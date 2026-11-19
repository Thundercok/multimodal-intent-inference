import XCTest
import CoreGraphics
@testable import GestureFlow

private final class MockContextProvider: DesktopContextProviderProtocol {
    var stubbedContext: DesktopContextRecord

    init(_ context: DesktopContextRecord) {
        self.stubbedContext = context
    }

    func currentContext() -> DesktopContextRecord {
        return stubbedContext
    }
}

final class ContextIntentFusionTests: XCTestCase {
    private func makeContext(
        bundleID: String = "com.microsoft.VSCode",
        category: String = "code_editor",
        role: String? = nil,
        windowState: String = "foreground",
        activeApp: Bool = true,
        focusedElement: Bool = false
    ) -> DesktopContextRecord {
        return DesktopContextRecord(
            activeAppBundleID: bundleID,
            appCategory: category,
            focusedElementRole: role,
            windowState: windowState,
            cursorPosition: [0.5, 0.5],
            timestamp: 1.0,
            availability: DesktopContextAvailability(
                activeApp: activeApp,
                focusedElement: focusedElement,
                cursorPosition: true,
                windowState: true
            )
        )
    }

    func testUnknownUnavailableContextActsAsExactUniformPassthrough() {
        let fusion = ContextIntentFusion(contextProvider: MockContextProvider(DesktopContextRecord()))
        let raw = IntentPrediction(
            intent: "pre_click",
            confidence: 0.68,
            isPreClickAnticipated: true,
            probabilities: [
                "idle": 0.12,
                "hover": 0.15,
                "pre_click": 0.68,
                "drag": 0.03,
                "scroll": 0.02
            ]
        )

        let fused = fusion.fuse(prediction: raw)
        XCTAssertEqual(fused.intent, "pre_click")
        XCTAssertTrue(fused.isPreClickAnticipated)
        XCTAssertEqual(fused.probabilities["pre_click"] ?? 0.0, 0.68, accuracy: 1e-4)
        XCTAssertEqual(fused.probabilities.values.reduce(0, +), 1.0, accuracy: 1e-4)
    }

    func testTypingFocusSuppressesAmbiguousPreClickWhilePreservingStrongPinch() {
        let typingCtx = makeContext(
            bundleID: "com.microsoft.VSCode",
            category: "code_editor",
            role: "AXTextArea",
            activeApp: true,
            focusedElement: true
        )
        let fusion = ContextIntentFusion(contextProvider: MockContextProvider(typingCtx))

        // Weak/ambiguous pre-click tremor during typing (0.68) -> suppressed to idle
        let ambiguous = IntentPrediction(
            intent: "pre_click",
            confidence: 0.68,
            isPreClickAnticipated: true,
            probabilities: [
                "idle": 0.25,
                "hover": 0.05,
                "pre_click": 0.68,
                "drag": 0.01,
                "scroll": 0.01
            ]
        )
        let suppressed = fusion.fuse(prediction: ambiguous)
        XCTAssertFalse(suppressed.isPreClickAnticipated, "Ambiguous pre-click jitter inside AXTextArea must be suppressed")
        XCTAssertEqual(suppressed.intent, "idle")
        XCTAssertEqual(suppressed.probabilities.values.reduce(0, +), 1.0, accuracy: 1e-4)

        // Deliberate strong visual clamp (0.95) -> still allowed through
        let deliberate = IntentPrediction(
            intent: "pre_click",
            confidence: 0.95,
            isPreClickAnticipated: true,
            probabilities: [
                "idle": 0.03,
                "hover": 0.01,
                "pre_click": 0.95,
                "drag": 0.005,
                "scroll": 0.005
            ]
        )
        let allowed = fusion.fuse(prediction: deliberate)
        XCTAssertTrue(allowed.isPreClickAnticipated, "Strong deliberate pre-click must override typing guard")
        XCTAssertGreaterThan(allowed.probabilities["pre_click"] ?? 0.0, 0.65)
    }

    func testBrowserScrollAreaBoostsAmbiguousScrollEvidence() {
        let browserScrollCtx = makeContext(
            bundleID: "com.apple.Safari",
            category: "browser",
            role: "AXScrollArea",
            activeApp: true,
            focusedElement: true
        )
        let fusion = ContextIntentFusion(contextProvider: MockContextProvider(browserScrollCtx))

        let borderline = IntentPrediction(
            intent: "hover",
            confidence: 0.45,
            isPreClickAnticipated: false,
            probabilities: [
                "idle": 0.08,
                "hover": 0.45,
                "pre_click": 0.04,
                "drag": 0.03,
                "scroll": 0.40
            ]
        )

        let fused = fusion.fuse(prediction: borderline)
        XCTAssertEqual(fused.intent, "scroll", "Safari + AXScrollArea context should resolve borderline hover/scroll in favor of scroll")
        XCTAssertGreaterThan(fused.probabilities["scroll"] ?? 0.0, fused.probabilities["hover"] ?? 1.0)
    }

    func testMacroGestureGuardBlocksAccidentalSwipesWhileTyping() {
        let typingCtx = makeContext(
            bundleID: "com.tinyspeck.slackmacgap",
            category: "communication",
            role: "AXTextField",
            activeApp: true,
            focusedElement: true
        )
        let fusion = ContextIntentFusion(contextProvider: MockContextProvider(typingCtx))

        XCTAssertFalse(fusion.shouldAllowMacroGesture(.swipe))
        XCTAssertFalse(fusion.shouldAllowMacroGesture(.pinchDock))
        XCTAssertFalse(fusion.shouldAllowMacroGesture(.spatialZoom))
        XCTAssertTrue(fusion.shouldAllowMacroGesture(.airCursor))

        // Without focused text element, macros remain unrestricted
        let navCtx = makeContext(bundleID: "com.apple.Safari", category: "browser", role: "AXScrollArea", activeApp: true, focusedElement: true)
        XCTAssertTrue(fusion.shouldAllowMacroGesture(.swipe, context: navCtx))
    }

    func testCursorEngineIntegrationWithContextFusion() {
        let typingCtx = makeContext(
            bundleID: "com.apple.dt.Xcode",
            category: "code_editor",
            role: "AXTextArea",
            activeApp: true,
            focusedElement: true
        )
        let mockProvider = MockContextProvider(typingCtx)
        let clock = ReplayClock()
        let sink = RecordingActionSink(clock: clock)
        let arbiter = IntentArbiter(clock: clock)
        let engine = CursorEngine(clock: clock, actionSink: sink, arbiter: arbiter, screenSize: CGSize(width: 1000, height: 1000))
        engine.useContextFusion = true
        engine.contextFusion = ContextIntentFusion(contextProvider: mockProvider)

        // Feed K=10 frames of static resting open palm so IntentClassifier sliding window populates
        for i in 0..<10 {
            var pts = Array(repeating: CGPoint.zero, count: 21)
            pts[0] = CGPoint(x: 0.50, y: 0.60)
            pts[4] = CGPoint(x: 0.39, y: 0.47)
            pts[5] = CGPoint(x: 0.46, y: 0.50)
            pts[8] = CGPoint(x: 0.46, y: 0.37)
            pts[9] = CGPoint(x: 0.50, y: 0.50)
            pts[12] = CGPoint(x: 0.50, y: 0.35)
            pts[13] = CGPoint(x: 0.54, y: 0.50)
            pts[16] = CGPoint(x: 0.54, y: 0.37)
            pts[17] = CGPoint(x: 0.57, y: 0.52)
            pts[20] = CGPoint(x: 0.57, y: 0.41)

            let lm = HandLandmarks(
                points: pts, fingerCount: 5, isTwoFingers: false, isCrossedFingers: false,
                isOneFinger: false, isOpenHand: true, isStrictOpenPalm: true,
                isInInteractionRange: true, tiltAngleDegrees: 0.0, isFist: false,
                pinchDistance: 0.12, isPinching: false, isFramingCorner: false, isThreeFingerPinch: false,
                thumbMiddleDistance: 0.16, indexMiddleDistance: 0.04,
                pinchCenter: pts[9], twoFingerCenter: pts[9], indexTip: pts[8], middleTip: pts[12],
                thumbTip: pts[4], wrist: pts[0], isRaisedHand: true, estimatedDepth: 0.40,
                depthVelocity: 0.0, isAirPush: false, featureTensor: [], handCentroid: pts[9],
                facingCamera: 0.90, faceBBox: nil, faceImpliedDepth: nil
            )
            clock.advance(to: 1.00 + Double(i) * 0.016)
            engine.process(lm)
        }

        XCTAssertEqual(engine.state, .pointing)
        XCTAssertFalse(engine.lastIntentPrediction.isPreClickAnticipated)
        let probSum = engine.lastIntentPrediction.probabilities.values.reduce(0, +)
        XCTAssertEqual(probSum, 1.0, accuracy: 1e-4)
    }
}
