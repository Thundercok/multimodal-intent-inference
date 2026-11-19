import Foundation

/// Closed-form O(1) Bayesian Multimodal Intent Fusion (Experiment 001).
///
/// Combines visual temporal sequence probabilities P(M_t | X) from `IntentClassifier`
/// with lightweight OS context priors pi(M_t | C_app, C_role) from `DesktopContextProvider`:
///   P(M_t | X, C) = normalize( P(M_t | X) * pi(M_t | C) )
///
/// Designed for ultra-low CPU/battery overhead (~microseconds per frame, zero neural weights).
final class ContextIntentFusion {
    static let shared = ContextIntentFusion()

    private static let typingRoles: Set<String> = [
        "AXTextField",
        "AXTextArea",
        "AXSecureTextField",
        "AXSearchField",
        "AXComboBox"
    ]

    private static let scrollRoles: Set<String> = [
        "AXScrollArea",
        "AXWebArea",
        "AXTable",
        "AXOutline",
        "AXList"
    ]

    private static let interactiveRoles: Set<String> = [
        "AXButton",
        "AXCheckBox",
        "AXRadioButton",
        "AXPopUpButton",
        "AXSlider",
        "AXMenuItem",
        "AXTabGroup"
    ]

    private let contextProvider: DesktopContextProviderProtocol

    init(contextProvider: DesktopContextProviderProtocol = DesktopContextProvider.shared) {
        self.contextProvider = contextProvider
    }

    static func isTypingContext(_ context: DesktopContextRecord) -> Bool {
        guard let role = context.focusedElementRole, context.availability.focusedElement else {
            return false
        }
        return typingRoles.contains(role)
    }

    /// Computes unnormalized multiplicative context prior pi(M_t | C) across the 5 intent classes.
    /// Defaults strictly to uniform [1, 1, 1, 1, 1] when context dimensions are unavailable.
    static func contextPriors(for context: DesktopContextRecord) -> [String: Float] {
        var priors: [String: Float] = [
            "idle": 1.0,
            "hover": 1.0,
            "pre_click": 1.0,
            "drag": 1.0,
            "scroll": 1.0
        ]

        // 1. App Category Modifiers (only if activeApp metadata is available)
        if context.availability.activeApp {
            switch AppCategory(rawValue: context.appCategory) ?? .unknown {
            case .browser:
                priors["scroll", default: 1.0] *= 1.25
                priors["pre_click", default: 1.0] *= 1.10
            case .codeEditor:
                priors["idle", default: 1.0] *= 1.15
                priors["scroll", default: 1.0] *= 1.15
                priors["drag", default: 1.0] *= 0.85
            case .communication:
                priors["idle", default: 1.0] *= 1.25
                priors["pre_click", default: 1.0] *= 0.85
                priors["drag", default: 1.0] *= 0.80
            case .media:
                priors["scroll", default: 1.0] *= 1.15
                priors["hover", default: 1.0] *= 0.90
                priors["drag", default: 1.0] *= 0.80
            case .fileManager:
                priors["drag", default: 1.0] *= 1.30
                priors["pre_click", default: 1.0] *= 1.15
            case .other, .unknown:
                break
            }
        }

        // 2. Focused UI Element Role Modifiers (privacy-safe AXRole only)
        if context.availability.focusedElement, let role = context.focusedElementRole {
            if typingRoles.contains(role) {
                priors["idle", default: 1.0] *= 2.40
                priors["hover", default: 1.0] *= 0.85
                priors["pre_click", default: 1.0] *= 0.45
                priors["drag", default: 1.0] *= 0.55
                priors["scroll", default: 1.0] *= 0.65
            } else if scrollRoles.contains(role) {
                priors["scroll", default: 1.0] *= 1.35
                priors["idle", default: 1.0] *= 0.90
                priors["hover", default: 1.0] *= 1.05
                priors["pre_click", default: 1.0] *= 1.05
                priors["drag", default: 1.0] *= 0.95
            } else if interactiveRoles.contains(role) {
                priors["pre_click", default: 1.0] *= 1.35
                priors["drag", default: 1.0] *= 1.15
                priors["hover", default: 1.0] *= 1.10
                priors["idle", default: 1.0] *= 0.85
                priors["scroll", default: 1.0] *= 0.80
            }
        }

        // 3. Window State Guard
        if context.availability.windowState && context.windowState == "hidden" {
            priors["idle", default: 1.0] *= 1.50
        }

        return priors
    }

    /// Fuses visual intent prediction with current cached OS context (or injected test context).
    func fuse(
        prediction: IntentPrediction,
        context: DesktopContextRecord? = nil,
        preClickThreshold: Float = 0.65
    ) -> IntentPrediction {
        let activeContext = context ?? contextProvider.currentContext()
        let priors = ContextIntentFusion.contextPriors(for: activeContext)

        var unnormalized: [String: Float] = [:]
        var total: Float = 0.0

        for cls in IntentClassifier.classes {
            let visualProb = max(0.0, prediction.probabilities[cls] ?? 0.0)
            let prior = priors[cls] ?? 1.0
            let joint = visualProb.isFinite ? (visualProb * prior) : 0.0
            unnormalized[cls] = joint
            total += joint
        }

        // Fallback cleanly if probability mass collapsed
        guard total > 1e-6 else {
            return prediction
        }

        var fusedProbs: [String: Float] = [:]
        var topClass = prediction.intent
        var topVal: Float = -1.0

        for cls in IntentClassifier.classes {
            let p = (unnormalized[cls] ?? 0.0) / total
            fusedProbs[cls] = p
            if p > topVal {
                topVal = p
                topClass = cls
            }
        }

        let pPreClick = fusedProbs["pre_click"] ?? 0.0
        let isPreClick = (pPreClick >= preClickThreshold) || (topClass == "pre_click" && !ContextIntentFusion.isTypingContext(activeContext))

        return IntentPrediction(
            intent: topClass,
            confidence: topVal,
            isPreClickAnticipated: isPreClick,
            probabilities: fusedProbs
        )
    }

    /// Returns whether a discrete gesture macro should proceed under current desktop context.
    func shouldAllowMacroGesture(_ gesture: GestureIntent, context: DesktopContextRecord? = nil) -> Bool {
        let activeContext = context ?? contextProvider.currentContext()
        guard ContextIntentFusion.isTypingContext(activeContext) else {
            return true
        }
        // While focused inside a text input / IDE editor, guard against accidental macro swipes
        // unless explicit scroll or pointer is active.
        switch gesture {
        case .swipe, .pinchDock, .spatialZoom:
            return false
        case .airScroll, .airCursor, .idle, .cooldown:
            return true
        }
    }
}
