import XCTest
import CoreGraphics
@testable import GestureFlow

final class DynamicFrictionPolicyTests: XCTestCase {
    func testPolicyModelLoadsSuccessfully() {
        let policy = DynamicFrictionPolicy()
        XCTAssertTrue(policy.isAvailable, "DynamicFrictionPolicy must load the bundled CoreML model")
    }

    func testObservationVectorDimensionsAndBounds() {
        let target = SnapTarget(id: "btn", rect: CGRect(x: 476, y: 476, width: 48, height: 48))
        let engine = MagneticSnappingEngine(targets: [target])
        // Approaching directly from the left along +X
        let cursor = CGPoint(x: 450, y: 500)
        let info = engine.nearestTargetInfo(for: cursor)
        XCTAssertNotNil(info)

        let obs = DynamicFrictionPolicy.buildObservation(
            velocityPx: CGVector(dx: 300.0, dy: 0.0),
            speedPxPerSec: 300.0,
            accelPxPerSec2: -1500.0,
            nearestTarget: info,
            aperture: 0.65
        )
        XCTAssertEqual(obs.count, 6)
        XCTAssertEqual(obs[0], 0.5, accuracy: 1e-4)   // 300 / 600
        XCTAssertEqual(obs[1], -0.5, accuracy: 1e-4)  // -1500 / 3000
        XCTAssertEqual(obs[2], 1.0, accuracy: 1e-4)   // within 2.5 * snapRadius
        XCTAssertGreaterThan(obs[3], 0.0)
        XCTAssertLessThan(obs[3], 1.0)
        XCTAssertEqual(obs[4], 1.0, accuracy: 1e-4)   // moving straight toward target (+X)
        XCTAssertEqual(obs[5], 0.65, accuracy: 1e-4)
    }

    func testPolicyOutputsStayWithinPhysicalActionBounds() {
        let policy = DynamicFrictionPolicy()
        XCTAssertTrue(policy.isAvailable)

        let target = SnapTarget(id: "btn", rect: CGRect(x: 476, y: 476, width: 48, height: 48))
        let info = MagneticSnappingEngine.NearestTargetInfo(target: target, distance: 35.0, dx: 35.0, dy: 0.0)

        let slowAction = policy.evaluate(
            velocityPx: CGVector(dx: 40.0, dy: 0.0),
            speedPxPerSec: 40.0,
            accelPxPerSec2: -800.0,
            nearestTarget: info,
            aperture: 0.55
        )
        XCTAssertTrue(slowAction.isLearned)
        XCTAssertGreaterThanOrEqual(slowAction.cutoffHz, DynamicFrictionPolicy.cutoffMin)
        XCTAssertLessThanOrEqual(slowAction.cutoffHz, DynamicFrictionPolicy.cutoffMax)
        XCTAssertGreaterThanOrEqual(slowAction.snapGain, DynamicFrictionPolicy.snapGainMin)
        XCTAssertLessThanOrEqual(slowAction.snapGain, DynamicFrictionPolicy.snapGainMax)

        let fastAction = policy.evaluate(
            velocityPx: CGVector(dx: 900.0, dy: 0.0),
            speedPxPerSec: 900.0,
            accelPxPerSec2: 1200.0,
            nearestTarget: nil,
            aperture: 0.78
        )
        XCTAssertTrue(fastAction.isLearned)
        XCTAssertGreaterThan(fastAction.cutoffHz, slowAction.cutoffHz, "High ballistic speed should yield higher cutoff than slow settling")
        XCTAssertGreaterThan(slowAction.snapGain, fastAction.snapGain, "Slow radial approach should engage higher snap gain than ballistic flight")
    }

    func testAnalyticalFallbackWhenModelUnavailable() {
        let policy = DynamicFrictionPolicy(modelURL: URL(fileURLWithPath: "/nonexistent/DynamicFrictionPolicy.mlmodel"), skipAutoDiscovery: true)
        XCTAssertFalse(policy.isAvailable)

        let actionSlow = policy.evaluate(velocityPx: .zero, speedPxPerSec: 50.0, accelPxPerSec2: 0.0, nearestTarget: nil, aperture: 0.75)
        XCTAssertFalse(actionSlow.isLearned)
        XCTAssertEqual(actionSlow.snapGain, 0.65, accuracy: 1e-4)

        let actionFast = policy.evaluate(velocityPx: CGVector(dx: 800, dy: 0), speedPxPerSec: 800.0, accelPxPerSec2: 0.0, nearestTarget: nil, aperture: 0.75)
        XCTAssertFalse(actionFast.isLearned)
        XCTAssertEqual(actionFast.snapGain, 0.0, accuracy: 1e-4)
    }

    func testCursorEngineIntegrationWithRLFrictionPolicy() {
        let clock = ReplayClock()
        let sink = RecordingActionSink(clock: clock)
        let arbiter = IntentArbiter(clock: clock)
        let engine = CursorEngine(clock: clock, actionSink: sink, arbiter: arbiter, screenSize: CGSize(width: 1000, height: 1000))
        engine.interactionMargin = 0.0
        engine.useRLFrictionPolicy = true
        engine.snappingEngine.registerTarget(SnapTarget(id: "ok", rect: CGRect(x: 476, y: 476, width: 48, height: 48)))

        let makeLM: (CGPoint) -> HandLandmarks = { mcp in
            var pts = Array(repeating: CGPoint.zero, count: 21)
            pts[0] = CGPoint(x: mcp.x, y: mcp.y + 0.35)
            pts[4] = CGPoint(x: mcp.x - 0.10, y: mcp.y - 0.05)
            pts[5] = CGPoint(x: mcp.x - 0.05, y: mcp.y)
            pts[8] = CGPoint(x: mcp.x, y: mcp.y - 0.10)
            pts[9] = mcp
            pts[12] = CGPoint(x: mcp.x, y: mcp.y - 0.15)
            pts[17] = CGPoint(x: mcp.x + 0.05, y: mcp.y)
            return HandLandmarks(
                points: pts, fingerCount: 4, isTwoFingers: false, isCrossedFingers: false,
                isOneFinger: false, isOpenHand: true, isStrictOpenPalm: true,
                isInInteractionRange: true, tiltAngleDegrees: 0.0, isFist: false,
                pinchDistance: 0.11, isPinching: false, isFramingCorner: false, isThreeFingerPinch: false,
                thumbMiddleDistance: 0.14, indexMiddleDistance: 0.05,
                pinchCenter: mcp, twoFingerCenter: mcp, indexTip: pts[8], middleTip: pts[12],
                thumbTip: pts[4], wrist: pts[0], isRaisedHand: true, estimatedDepth: 0.40,
                depthVelocity: 0.0, isAirPush: false, featureTensor: [], handCentroid: mcp,
                facingCamera: 0.90, faceBBox: nil, faceImpliedDepth: nil
            )
        }

        clock.advance(to: 1.00)
        engine.process(makeLM(CGPoint(x: 0.56, y: 0.50)))
        clock.advance(to: 1.016)
        engine.process(makeLM(CGPoint(x: 0.52, y: 0.50)))

        XCTAssertEqual(engine.state, .pointing)
        XCTAssertNotNil(engine.lastSnapGain)
        XCTAssertGreaterThanOrEqual(engine.currentCutoff, DynamicFrictionPolicy.cutoffMin)
        XCTAssertLessThanOrEqual(engine.currentCutoff, DynamicFrictionPolicy.cutoffMax)
    }
}
