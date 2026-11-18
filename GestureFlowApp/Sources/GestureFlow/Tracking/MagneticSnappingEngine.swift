import Foundation
import CoreGraphics

/// Target representation for magnetic cursor attraction in screen space
public struct SnapTarget: Equatable {
    public let id: String
    public let rect: CGRect
    public var center: CGPoint {
        CGPoint(x: rect.midX, y: rect.midY)
    }
    public var snapRadius: CGFloat
    public var strength: CGFloat // Attraction gain k in [0.0 ... 1.0], default 0.65

    public init(id: String, rect: CGRect, snapRadius: CGFloat? = nil, strength: CGFloat = 0.65) {
        self.id = id
        self.rect = rect
        let halfDiag = hypot(rect.width, rect.height) * 0.5
        self.snapRadius = snapRadius ?? max(30.0, halfDiag + 25.0)
        self.strength = max(0.0, min(1.0, strength))
    }
}

/// Adaptive Magnetic Snapping Engine (Module 2 of RESEARCH_ROADMAP.md)
///
/// Implements a continuous Gaussian potential well around UI targets:
///   F_attract = k * (c_target - p_cursor) * exp(-r^2 / (2 * sigma^2))
///
/// Features:
/// 1. C^infinity smooth potential well: zero cursor teleportation or discontinuity.
/// 2. Sticky-Trap Breakaway: Modulates attraction by cursor speed. When user makes a ballistic flick
///    (speed > breakawayThreshold), the attraction smoothly drops to 0, eliminating sticky trapping.
/// 3. Nearest-target selection: avoids saddle point artifacts when two buttons are adjacent.
public final class MagneticSnappingEngine {
    public var targets: [SnapTarget] = []
    public var isEnabled: Bool = true

    /// Breakaway parameters (in px/sec)
    /// Below lowSpeedThreshold, full magnetic attraction is exerted.
    /// Above breakawayThreshold, magnetic attraction is completely disengaged.
    public var lowSpeedThreshold: CGFloat = 200.0   // px/sec
    public var breakawayThreshold: CGFloat = 650.0  // px/sec

    public init(targets: [SnapTarget] = [], isEnabled: Bool = true) {
        self.targets = targets
        self.isEnabled = isEnabled
    }

    /// Register or update a snap target
    public func registerTarget(_ target: SnapTarget) {
        targets.removeAll { $0.id == target.id }
        targets.append(target)
    }

    /// Remove a snap target by ID
    public func unregisterTarget(id: String) {
        targets.removeAll { $0.id == id }
    }

    /// Clear all targets
    public func clearTargets() {
        targets.removeAll()
    }

    /// Target geometry context for policy observation (zero oracle knowledge)
    public struct NearestTargetInfo {
        public let target: SnapTarget
        public let distance: CGFloat
        public let dx: CGFloat
        public let dy: CGFloat
    }

    /// Returns nearest registered target within active snap basin (`2.5 * snapRadius`), if any.
    public func nearestTargetInfo(for cursorPos: CGPoint) -> NearestTargetInfo? {
        guard isEnabled, !targets.isEmpty else {
            return nil
        }
        var bestTarget: SnapTarget?
        var minDistance: CGFloat = .infinity
        var bestDx: CGFloat = 0.0
        var bestDy: CGFloat = 0.0

        for target in targets {
            let center = target.center
            let dx = center.x - cursorPos.x
            let dy = center.y - cursorPos.y
            let dist = hypot(dx, dy)
            if dist < target.snapRadius * 2.5 && dist < minDistance {
                minDistance = dist
                bestTarget = target
                bestDx = dx
                bestDy = dy
            }
        }

        guard let target = bestTarget else { return nil }
        return NearestTargetInfo(target: target, distance: minDistance, dx: bestDx, dy: bestDy)
    }

    /// Calculate snapped cursor position given raw cursor coordinate, velocity (in px/sec),
    /// and optional RL policy snap gain override (`snapGainOverride` in `[0, 0.85]`).
    public func applySnapping(
        cursorPos: CGPoint,
        speedPxPerSec: CGFloat,
        snapGainOverride: CGFloat? = nil
    ) -> CGPoint {
        guard isEnabled, !targets.isEmpty else {
            return cursorPos
        }

        // 1. Resolve nearest target once (early return before checking velocity when no targets nearby)
        let targetInfo = nearestTargetInfo(for: cursorPos)
        guard let info = targetInfo, info.distance >= 0.001 else {
            return cursorPos
        }

        // 2. Determine effective snap gain (RL override or Hermite velocity breakaway heuristic)
        let effectiveGain: CGFloat
        if let overrideGain = snapGainOverride {
            effectiveGain = max(0.0, min(0.85, overrideGain))
        } else {
            let velocityWeight: CGFloat
            if speedPxPerSec <= lowSpeedThreshold {
                velocityWeight = 1.0
            } else if speedPxPerSec >= breakawayThreshold {
                velocityWeight = 0.0
            } else {
                let t = (speedPxPerSec - lowSpeedThreshold) / (breakawayThreshold - lowSpeedThreshold)
                // Smoothstep hermite interpolation: 1 - (3t^2 - 2t^3)
                velocityWeight = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
            }
            effectiveGain = info.target.strength * velocityWeight
        }

        if effectiveGain <= 0.001 {
            return cursorPos
        }

        // 3. Gaussian attraction potential
        let sigma = info.target.snapRadius * 0.707 // sigma chosen so peak pull is within target margin
        let gaussian = exp(-CGFloat(info.distance * info.distance) / (2.0 * sigma * sigma))
        let pullFactor = effectiveGain * gaussian

        let snappedX = cursorPos.x + info.dx * pullFactor
        let snappedY = cursorPos.y + info.dy * pullFactor

        return CGPoint(x: snappedX, y: snappedY)
    }
}
