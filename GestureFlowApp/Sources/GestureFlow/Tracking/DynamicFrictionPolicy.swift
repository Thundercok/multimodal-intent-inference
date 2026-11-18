import Foundation
import CoreGraphics
import CoreML

struct FrictionControlAction: Equatable {
    let cutoffHz: Double
    let snapGain: CGFloat
    let isLearned: Bool
}

/// Phase 3: PPO-Trained Dynamic Viscous Friction & Magnetic Snapping Policy (Module 2 of RESEARCH_ROADMAP.md)
///
/// Observes the 6D target-agnostic kinematic/proximity state:
///   0: norm_speed        = min(2.0, speedPxPerSec / 600.0)
///   1: norm_signed_accel = clamp(accelPxPerSec2 / 3000.0, -2.0, 2.0)
///   2: has_target_prox   = 1.0 if nearest target within 2.5 * snapRadius else 0.0
///   3: norm_target_dist  = min(1.0, dist / (2.5 * snapRadius))
///   4: radial_alignment  = cos(theta(dir_to_nearest, velocity)) in [-1, 1]
///   5: aperture_ratio    = clamp(aperture, 0.0, 1.0)
///
/// Outputs continuous optimal control actions:
///   - cutoffHz in [0.25, 4.00] Hz (adaptive 1€ low-speed tremor filter vs high-speed ballistic glide)
///   - snapGain in [0.00, 0.85]    (target-direction-aware Gaussian pull / distractor breakaway)
final class DynamicFrictionPolicy {
    static let shared = DynamicFrictionPolicy()

    static let stateDim = 6
    static let cutoffMin: Double = 0.25
    static let cutoffMax: Double = 4.00
    static let snapGainMin: CGFloat = 0.00
    static let snapGainMax: CGFloat = 0.85

    private var model: MLModel? = nil
    private(set) var isAvailable: Bool = false

    init(modelURL: URL? = nil, skipAutoDiscovery: Bool = false) {
        if let url = modelURL {
            loadModel(at: url)
        } else if !skipAutoDiscovery {
            discoverAndLoadModel()
        }
    }

    private func discoverAndLoadModel() {
        if let resourceURL = Bundle.main.url(forResource: "DynamicFrictionPolicy", withExtension: "mlmodelc") {
            loadModel(at: resourceURL)
            return
        }
        if let resourceURL = Bundle.main.url(forResource: "DynamicFrictionPolicy", withExtension: "mlmodel") {
            loadModel(at: resourceURL)
            return
        }

        let candidatePaths = [
            "Resources/DynamicFrictionPolicy.mlmodelc",
            "Resources/DynamicFrictionPolicy.mlmodel",
            "../Resources/DynamicFrictionPolicy.mlmodelc",
            "../Resources/DynamicFrictionPolicy.mlmodel",
            "GestureFlowApp/Resources/DynamicFrictionPolicy.mlmodelc",
            "GestureFlowApp/Resources/DynamicFrictionPolicy.mlmodel",
            "Sources/GestureFlow/Resources/DynamicFrictionPolicy.mlmodelc",
            "Sources/GestureFlow/Resources/DynamicFrictionPolicy.mlmodel"
        ]
        for path in candidatePaths {
            let url = URL(fileURLWithPath: path)
            if FileManager.default.fileExists(atPath: url.path) {
                loadModel(at: url)
                return
            }
        }
    }

    private func loadModel(at url: URL) {
        do {
            let compiledURL: URL
            if url.pathExtension == "mlmodel" {
                compiledURL = try MLModel.compileModel(at: url)
            } else {
                compiledURL = url
            }
            let config = MLModelConfiguration()
            config.computeUnits = .all
            self.model = try MLModel(contentsOf: compiledURL, configuration: config)
            self.isAvailable = true
        } catch {
            self.model = nil
            self.isAvailable = false
        }
    }

    /// Constructs the 6D target-agnostic observation vector matching Python PPO training.
    static func buildObservation(
        velocityPx: CGVector,
        speedPxPerSec: Double,
        accelPxPerSec2: Double,
        nearestTarget: MagneticSnappingEngine.NearestTargetInfo?,
        aperture: CGFloat
    ) -> [Float] {
        let normSpeed = Float(min(2.0, max(0.0, speedPxPerSec / 600.0)))
        let normAccel = Float(max(-2.0, min(2.0, accelPxPerSec2 / 3000.0)))

        let hasTargetProx: Float
        let normTargetDist: Float
        let radialAlignment: Float

        if let info = nearestTarget {
            hasTargetProx = 1.0
            let basin = max(1.0, info.target.snapRadius * 2.5)
            normTargetDist = Float(min(1.0, max(0.0, info.distance / basin)))
            let vNorm = hypot(velocityPx.dx, velocityPx.dy)
            if info.distance > 1e-3 && vNorm > 1e-3 {
                let dot = info.dx * velocityPx.dx + info.dy * velocityPx.dy
                radialAlignment = Float(max(-1.0, min(1.0, dot / (info.distance * vNorm))))
            } else {
                radialAlignment = 0.0
            }
        } else {
            hasTargetProx = 0.0
            normTargetDist = 1.0
            radialAlignment = 0.0
        }

        let apRatio = Float(max(0.0, min(1.0, aperture)))
        return [normSpeed, normAccel, hasTargetProx, normTargetDist, radialAlignment, apRatio]
    }

    /// Evaluates optimal control actions `(cutoffHz, snapGain)` using CoreML PPO actor or analytical fallback.
    func evaluate(
        velocityPx: CGVector,
        speedPxPerSec: Double,
        accelPxPerSec2: Double,
        nearestTarget: MagneticSnappingEngine.NearestTargetInfo?,
        aperture: CGFloat
    ) -> FrictionControlAction {
        guard let model = self.model else {
            return analyticalFallback(speedPxPerSec: speedPxPerSec)
        }

        let obs = DynamicFrictionPolicy.buildObservation(
            velocityPx: velocityPx,
            speedPxPerSec: speedPxPerSec,
            accelPxPerSec2: accelPxPerSec2,
            nearestTarget: nearestTarget,
            aperture: aperture
        )

        do {
            let mlArray = try MLMultiArray(shape: [1, NSNumber(value: DynamicFrictionPolicy.stateDim)], dataType: .float32)
            for (idx, val) in obs.enumerated() {
                mlArray[idx] = NSNumber(value: val)
            }

            let inputProvider = try MLDictionaryFeatureProvider(dictionary: ["obs_features": mlArray])
            let output = try model.prediction(from: inputProvider)

            if let rawOut = output.featureValue(for: "action_tanh")?.multiArrayValue, rawOut.count >= 2 {
                let u0 = max(-1.0, min(1.0, Double(rawOut[0].floatValue)))
                let u1 = max(-1.0, min(1.0, Double(rawOut[1].floatValue)))

                let cutoff = DynamicFrictionPolicy.cutoffMin +
                    (DynamicFrictionPolicy.cutoffMax - DynamicFrictionPolicy.cutoffMin) * 0.5 * (u0 + 1.0)
                let snapGain = DynamicFrictionPolicy.snapGainMin +
                    (DynamicFrictionPolicy.snapGainMax - DynamicFrictionPolicy.snapGainMin) * CGFloat(0.5 * (u1 + 1.0))

                return FrictionControlAction(cutoffHz: cutoff, snapGain: snapGain, isLearned: true)
            }
        } catch {
            return analyticalFallback(speedPxPerSec: speedPxPerSec)
        }

        return analyticalFallback(speedPxPerSec: speedPxPerSec)
    }

    func analyticalFallback(speedPxPerSec: Double) -> FrictionControlAction {
        // Fallback matching GestureFlow quadratic viscous cutoff + Hermite breakaway
        let normAnchorSpeed = speedPxPerSec / 1440.0
        let linearRatio = min(1.0, max(0.0, normAnchorSpeed / 0.55))
        let cutoff = 0.40 + (2.50 - 0.40) * (linearRatio * linearRatio)

        let snapGain: CGFloat
        if speedPxPerSec <= 200.0 {
            snapGain = 0.65
        } else if speedPxPerSec >= 650.0 {
            snapGain = 0.0
        } else {
            let t = CGFloat((speedPxPerSec - 200.0) / (650.0 - 200.0))
            snapGain = 0.65 * (1.0 - (3.0 * t * t - 2.0 * t * t * t))
        }
        return FrictionControlAction(cutoffHz: cutoff, snapGain: snapGain, isLearned: false)
    }
}
