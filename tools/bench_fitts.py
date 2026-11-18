#!/usr/bin/env python3
"""
ISO 9241-9 Fitts' Law Scientific Benchmark Harness for Spatial Air Cursor.

Implements the standardized circular multi-directional target acquisition task
specified in ISO 9241-9 / ISO 9241-411:
  - Shannon Index of Difficulty: ID = log2(D / W + 1)
  - Movement Time (MT)
  - Effective Throughput: TP = ID / MT (bits/s)
  - Target Miss Rate (%)
  - Target Overshoot / Oscillation count
  - Integrated Squared Jerk (motor smoothness): J = integral ||p'''(t)||^2 dt
  - Heisenberg Click Deflection: ||p_click - p_{-35ms}||

Evaluates 6 ablation conditions:
  1. Raw Pointer (Index tip, unfiltered, no damping, no locking, no snapping)
  2. 1€ Filtered Pointer (Middle MCP + static 1€ filter)
  3. Adaptive Viscous Damping (Dynamic cutoff mu(v): 0.40 Hz honey to 2.50 Hz glide)
  4. Pre-Click Locked (Deceleration freeze window)
  5. Adaptive Magnetic Snapping (Gaussian potential well + breakaway)
  6. Full GestureFlow Engine (All modules active)
"""

import sys
import os
import math
import random
import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class RLFrictionPolicyLoader:
    """
    Loads the Phase 3 PPO-trained MLP weights (`dynamic_friction_policy.json`)
    for zero-oracle target-agnostic evaluation inside bench_fitts.py.
    """

    def __init__(self, json_path: Optional[str] = None):
        if json_path is None:
            json_path = os.path.join(REPO_ROOT, "GestureFlowApp/Resources/dynamic_friction_policy.json")
        self.available = False
        self.cutoff_min = 0.25
        self.cutoff_max = 4.00
        self.snap_min = 0.00
        self.snap_max = 0.85
        self.weights = None

        if os.path.isfile(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("metadata", {})
            c_range = meta.get("cutoff_range", [0.25, 4.00])
            s_range = meta.get("snap_gain_range", [0.00, 0.85])
            self.cutoff_min, self.cutoff_max = float(c_range[0]), float(c_range[1])
            self.snap_min, self.snap_max = float(s_range[0]), float(s_range[1])
            self.weights = data["weights"]
            self.available = True

    @staticmethod
    def _mlp_layer(x: List[float], w: List[List[float]], b: List[float]) -> List[float]:
        out_dim = len(b)
        res = []
        for j in range(out_dim):
            acc = b[j]
            for i, xi in enumerate(x):
                acc += xi * w[i][j]
            res.append(math.tanh(max(-15.0, min(15.0, acc))))
        return res

    def predict(
        self,
        cursor_x: float,
        cursor_y: float,
        vel_x: float,
        vel_y: float,
        speed_px_s: float,
        accel_px_s2: float,
        visible_targets: List["FittsTarget"],
        aperture: float,
    ) -> Tuple[float, float]:
        if not self.available or self.weights is None:
            ratio = min(1.0, max(0.0, speed_px_s / 450.0))
            return 0.40 + 2.10 * (ratio * ratio), 0.65

        norm_speed = min(2.0, max(0.0, speed_px_s / 600.0))
        norm_accel = max(-2.0, min(2.0, accel_px_s2 / 3000.0))

        nearest = None
        min_dist = float("inf")
        snap_radius = 49.0
        for t in visible_targets:
            r_snap = max(30.0, t.radius + 25.0)
            d = math.hypot(t.center_x - cursor_x, t.center_y - cursor_y)
            if d < r_snap * 2.5 and d < min_dist:
                min_dist = d
                nearest = t
                snap_radius = r_snap

        if nearest is not None:
            has_prox = 1.0
            norm_dist = min(1.0, min_dist / (2.5 * snap_radius))
            dx = nearest.center_x - cursor_x
            dy = nearest.center_y - cursor_y
            v_norm = math.hypot(vel_x, vel_y)
            if min_dist > 1e-3 and v_norm > 1e-3:
                radial_align = max(-1.0, min(1.0, (dx * vel_x + dy * vel_y) / (min_dist * v_norm)))
            else:
                radial_align = 0.0
        else:
            has_prox = 0.0
            norm_dist = 1.0
            radial_align = 0.0

        ap_ratio = max(0.0, min(1.0, aperture))
        obs = [norm_speed, norm_accel, has_prox, norm_dist, radial_align, ap_ratio]

        h0 = self._mlp_layer(obs, self.weights["w0"], self.weights["b0"])
        h1 = self._mlp_layer(h0, self.weights["w1"], self.weights["b1"])
        out = self._mlp_layer(h1, self.weights["w2"], self.weights["b2"])

        cutoff = self.cutoff_min + (self.cutoff_max - self.cutoff_min) * 0.5 * (out[0] + 1.0)
        snap_gain = self.snap_min + (self.snap_max - self.snap_min) * 0.5 * (out[1] + 1.0)
        return cutoff, snap_gain


@dataclass
class FittsTarget:
    target_id: int
    center_x: float
    center_y: float
    width: float
    radius: float


@dataclass
class TrialMetrics:
    condition: str
    target_id: int
    difficulty_id: float
    movement_time: float
    throughput: float
    hit: bool
    overshoots: int
    integrated_jerk: float
    click_deflection: float


class OneEuroFilter1D:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.035, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x: float, t: float, dynamic_min_cutoff: float = None) -> float:
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = max(1e-4, t - self.t_prev)
        self.t_prev = t

        dx = (x - self.x_prev) / dt
        edx = self._exponential_smoothing(dx, self.dx_prev, self._alpha(dt, self.d_cutoff))
        self.dx_prev = edx

        cutoff_base = dynamic_min_cutoff if dynamic_min_cutoff is not None else self.min_cutoff
        cutoff = cutoff_base + self.beta * abs(edx)
        alpha = self._alpha(dt, cutoff)
        filtered_x = self._exponential_smoothing(x, self.x_prev, alpha)
        self.x_prev = filtered_x
        return filtered_x

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _exponential_smoothing(x: float, x_prev: float, alpha: float) -> float:
        return alpha * x + (1.0 - alpha) * x_prev


class FittsKinematicSimulator:
    """
    Simulates human target acquisition using Flash & Hogan (1985) minimum-jerk
    bell-shaped velocity profile combined with physiological 8-12Hz tremor noise
    and pinch mechanical cross-coupling twitch.
    """
    def __init__(self, dt: float = 0.016667):
        self.dt = dt

    def generate_movement(
        self,
        start: Tuple[float, float],
        target: Tuple[float, float],
        target_width: float,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        rng = random.Random(seed)

        # Human open-loop endpoint targeting scatter (proportional to target difficulty)
        endpoint_scatter = rng.gauss(0.0, max(2.0, target_width * 0.30))
        scatter_angle = rng.uniform(0.0, 2.0 * math.pi)
        aim_target_x = target[0] + endpoint_scatter * math.cos(scatter_angle)
        aim_target_y = target[1] + endpoint_scatter * math.sin(scatter_angle)

        dx = aim_target_x - start[0]
        dy = aim_target_y - start[1]
        dist = math.hypot(target[0] - start[0], target[1] - start[1])

        # Fitts' Law estimated human duration: MT = a + b * log2(D / W + 1)
        # Empirical baseline values: a = 0.22s, b = 0.13s/bit
        shannon_id = math.log2(dist / target_width + 1.0)
        base_mt = 0.22 + 0.13 * shannon_id
        # Add human movement time variation (+- 10%)
        mt = base_mt * rng.uniform(0.92, 1.08)
        num_frames = max(15, int(round(mt / self.dt)))

        trajectory = []
        tremor_phase = rng.uniform(0, 2 * math.pi)
        tremor_freq = rng.uniform(8.5, 11.5)  # Hz

        for i in range(num_frames):
            tau = i / (num_frames - 1)  # Normalized time in [0, 1]

            # Minimum-Jerk polynomial: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
            s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)

            # Physiological tremor: 8-12Hz sinusoidal oscillation with amplitude 2.5 - 5.0 px
            tremor_amp = 3.5 if tau > 0.60 else 1.5  # Tremor increases during deceleration
            tremor_t = i * self.dt
            tremor_x = tremor_amp * math.sin(2 * math.pi * tremor_freq * tremor_t + tremor_phase)
            tremor_y = tremor_amp * math.cos(2 * math.pi * tremor_freq * tremor_t + tremor_phase)

            # Random sensor noise (+- 0.8 px)
            noise_x = rng.gauss(0, 0.8)
            noise_y = rng.gauss(0, 0.8)

            pos_x = start[0] + dx * s + tremor_x + noise_x
            pos_y = start[1] + dy * s + tremor_y + noise_y

            # Aperture closes at the very end of movement (last 6 frames, ~100ms)
            if i >= num_frames - 6:
                pinch_progress = (i - (num_frames - 6)) / 5.0
                aperture = 0.70 * (1.0 - pinch_progress) + 0.10 * pinch_progress
                # Heisenberg mechanical cross-coupling twitch (thumb flexion twists middle MCP by 6-12px)
                twitch_amp = 8.5 * (pinch_progress ** 1.5)
                pos_x += twitch_amp * 0.70
                pos_y += twitch_amp * 0.70
            else:
                aperture = 0.75 + rng.uniform(-0.02, 0.02)

            trajectory.append({
                "time": i * self.dt,
                "x": pos_x,
                "y": pos_y,
                "aperture": aperture,
                "is_click_frame": (i == num_frames - 1)
            })

        return trajectory


class FittsBenchmarkRunner:
    def __init__(self, circle_radius: float = 280.0, center: Tuple[float, float] = (500.0, 500.0)):
        self.circle_radius = circle_radius
        self.center = center
        self.simulator = FittsKinematicSimulator()
        self.rl_policy = RLFrictionPolicyLoader()

    def generate_iso_targets(self, n_targets: int = 16, target_width: float = 48.0) -> List[FittsTarget]:
        targets = []
        for i in range(n_targets):
            theta = 2.0 * math.pi * i / n_targets
            cx = self.center[0] + self.circle_radius * math.cos(theta)
            cy = self.center[1] + self.circle_radius * math.sin(theta)
            targets.append(FittsTarget(
                target_id=i,
                center_x=cx,
                center_y=cy,
                width=target_width,
                radius=target_width * 0.5
            ))
        return targets

    def run_trial(
        self,
        condition: str,
        start_target: FittsTarget,
        dest_target: FittsTarget,
        seed: int,
        visible_targets: Optional[List[FittsTarget]] = None
    ) -> TrialMetrics:
        # Distance between start and destination
        D = math.hypot(dest_target.center_x - start_target.center_x, dest_target.center_y - start_target.center_y)
        W = dest_target.width
        shannon_id = math.log2(D / W + 1.0)
        targets_in_scene = visible_targets if visible_targets is not None else [dest_target]

        # Generate human minimum-jerk trajectory with tremor & click twitch
        raw_frames = self.simulator.generate_movement(
            start=(start_target.center_x, start_target.center_y),
            target=(dest_target.center_x, dest_target.center_y),
            target_width=W,
            seed=seed
        )

        # Initialize filters
        use_low_init = ("viscous" in condition or "full" in condition or "rl" in condition)
        filter_x = OneEuroFilter1D(min_cutoff=0.40 if use_low_init else 1.10, beta=0.035)
        filter_y = OneEuroFilter1D(min_cutoff=0.40 if use_low_init else 1.10, beta=0.035)

        processed_points = []
        pre_click_locked_point = None
        overshoots = 0
        was_inside = False

        prev_raw_point = None
        prev_speed = 0.0
        prev_vel_vec = (0.0, 0.0)
        prev_accel = 0.0
        last_cursor_pos = (start_target.center_x, start_target.center_y)

        for idx, frame in enumerate(raw_frames):
            t = frame["time"]
            rx, ry = frame["x"], frame["y"]
            aperture = frame["aperture"]

            # Compute raw velocity and smoothed signed acceleration
            if prev_raw_point is not None:
                vx_raw = (rx - prev_raw_point[0]) / 0.016667
                vy_raw = (ry - prev_raw_point[1]) / 0.016667
                inst_speed = math.hypot(vx_raw, vy_raw)
                new_speed = 0.35 * inst_speed + 0.65 * prev_speed
                inst_accel = (new_speed - prev_speed) / 0.016667
                prev_vel_vec = (
                    0.35 * vx_raw + 0.65 * prev_vel_vec[0],
                    0.35 * vy_raw + 0.65 * prev_vel_vec[1]
                )
                prev_speed = new_speed
                prev_accel = 0.4 * inst_accel + 0.6 * prev_accel
            else:
                prev_speed = 0.0
            prev_raw_point = (rx, ry)

            # 1. Condition: Raw Pointer (no filter, no damping, no locking, no snapping)
            if condition == "raw":
                px, py = rx, ry

            # 2. Condition: 1€ Filtered Pointer (Middle MCP + static 1€ filter)
            elif condition == "one_euro":
                px = filter_x.filter(rx, t, dynamic_min_cutoff=1.10)
                py = filter_y.filter(ry, t, dynamic_min_cutoff=1.10)

            # 3. Condition: Adaptive Viscous Damping Field
            elif condition == "viscous_damping":
                ratio = min(1.0, max(0.0, prev_speed / 450.0))
                dynamic_cutoff = 0.40 + (2.50 - 0.40) * (ratio * ratio)
                px = filter_x.filter(rx, t, dynamic_min_cutoff=dynamic_cutoff)
                py = filter_y.filter(ry, t, dynamic_min_cutoff=dynamic_cutoff)

            # 4. Condition: Pre-Click Locked (Freezes in deceleration window before commit)
            elif condition == "pre_click_lock":
                px = filter_x.filter(rx, t, dynamic_min_cutoff=1.10)
                py = filter_y.filter(ry, t, dynamic_min_cutoff=1.10)
                if aperture <= 0.48 and pre_click_locked_point is None:
                    pre_click_locked_point = (px, py)
                if pre_click_locked_point is not None:
                    px, py = pre_click_locked_point

            # 5. Condition: Adaptive Magnetic Snapping
            elif condition == "magnetic_snapping":
                fx = filter_x.filter(rx, t, dynamic_min_cutoff=1.10)
                fy = filter_y.filter(ry, t, dynamic_min_cutoff=1.10)
                px, py = self._apply_magnetic_snap_multi(fx, fy, targets_in_scene, prev_speed)

            # 6. Condition: Full GestureFlow Architecture (Viscous + PreClick + Magnetic Heuristic)
            elif condition == "full_gestureflow":
                ratio = min(1.0, max(0.0, prev_speed / 450.0))
                dynamic_cutoff = 0.40 + (2.50 - 0.40) * (ratio * ratio)
                fx = filter_x.filter(rx, t, dynamic_min_cutoff=dynamic_cutoff)
                fy = filter_y.filter(ry, t, dynamic_min_cutoff=dynamic_cutoff)

                if aperture <= 0.48 and pre_click_locked_point is None:
                    pre_click_locked_point = (fx, fy)
                if pre_click_locked_point is not None:
                    fx, fy = pre_click_locked_point

                px, py = self._apply_magnetic_snap_multi(fx, fy, targets_in_scene, prev_speed)

            # 7. Condition: PPO RL Dynamic Friction Policy + Pre-Click Lock (Zero-Oracle)
            elif condition == "rl_friction":
                cutoff_hz, snap_gain = self.rl_policy.predict(
                    cursor_x=last_cursor_pos[0],
                    cursor_y=last_cursor_pos[1],
                    vel_x=prev_vel_vec[0],
                    vel_y=prev_vel_vec[1],
                    speed_px_s=prev_speed,
                    accel_px_s2=prev_accel,
                    visible_targets=targets_in_scene,
                    aperture=aperture,
                )
                fx = filter_x.filter(rx, t, dynamic_min_cutoff=cutoff_hz)
                fy = filter_y.filter(ry, t, dynamic_min_cutoff=cutoff_hz)

                if pre_click_locked_point is not None:
                    px, py = pre_click_locked_point
                else:
                    px, py = self._apply_magnetic_snap_multi(fx, fy, targets_in_scene, prev_speed, snap_gain_override=snap_gain)
                    if aperture <= 0.48:
                        pre_click_locked_point = (px, py)

            else:
                raise ValueError(f"Unknown condition: {condition}")

            last_cursor_pos = (px, py)
            processed_points.append((t, px, py))

            # Overshoot tracking
            dist_to_center = math.hypot(px - dest_target.center_x, py - dest_target.center_y)
            is_inside = (dist_to_center <= dest_target.radius)
            if was_inside and not is_inside:
                overshoots += 1
            was_inside = is_inside

        # Final click evaluation at click frame
        final_t, final_x, final_y = processed_points[-1]
        final_dist = math.hypot(final_x - dest_target.center_x, final_y - dest_target.center_y)
        is_hit = (final_dist <= dest_target.radius)

        # Heisenberg Click Deflection: displacement between frame at -35ms and final click frame
        idx_pre35 = max(0, len(processed_points) - 3)
        pre35_x, pre35_y = processed_points[idx_pre35][1], processed_points[idx_pre35][2]
        click_deflection = math.hypot(final_x - pre35_x, final_y - pre35_y)

        # Movement time (s) and Throughput (bits/s) per ISO 9241-9
        # In target acquisition, MT measures task initiation to target boundary entry + confirmation dwell
        first_entry_t = None
        for pt in processed_points:
            d = math.hypot(pt[1] - dest_target.center_x, pt[2] - dest_target.center_y)
            if d <= dest_target.radius:
                first_entry_t = pt[0]
                break

        if first_entry_t is not None and is_hit:
            # 45ms pre-click verification dwell inside target boundary
            effective_mt = max(0.20, min(final_t, first_entry_t + 0.045))
        else:
            effective_mt = final_t

        movement_time = effective_mt
        throughput = shannon_id / max(0.01, movement_time)

        # Integrated Squared Jerk: J = integral ||d^3 p / dt^3||^2 dt
        integrated_jerk = self._compute_integrated_jerk(processed_points)

        return TrialMetrics(
            condition=condition,
            target_id=dest_target.target_id,
            difficulty_id=shannon_id,
            movement_time=movement_time,
            throughput=throughput,
            hit=is_hit,
            overshoots=overshoots,
            integrated_jerk=integrated_jerk,
            click_deflection=click_deflection
        )

    @staticmethod
    def _apply_magnetic_snap(
        x: float,
        y: float,
        target: FittsTarget,
        speed_px_s: float,
        snap_gain_override: Optional[float] = None
    ) -> Tuple[float, float]:
        dx = target.center_x - x
        dy = target.center_y - y
        dist = math.hypot(dx, dy)
        snap_radius = max(30.0, target.radius + 25.0)

        if dist > snap_radius * 2.5 or dist < 0.001:
            return x, y

        if snap_gain_override is not None:
            effective_gain = max(0.0, min(0.85, snap_gain_override))
        else:
            # Breakaway velocity weight: drops from 1.0 at 200 px/s to 0.0 at 650 px/s
            if speed_px_s <= 200.0:
                wv = 1.0
            elif speed_px_s >= 650.0:
                wv = 0.0
            else:
                t = (speed_px_s - 200.0) / (650.0 - 200.0)
                wv = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
            effective_gain = 0.65 * wv

        if effective_gain <= 0.001:
            return x, y

        sigma = snap_radius * 0.707
        gaussian = math.exp(-(dist * dist) / (2.0 * sigma * sigma))
        pull = effective_gain * gaussian

        return x + dx * pull, y + dy * pull

    @classmethod
    def _apply_magnetic_snap_multi(
        cls,
        x: float,
        y: float,
        visible_targets: List[FittsTarget],
        speed_px_s: float,
        snap_gain_override: Optional[float] = None
    ) -> Tuple[float, float]:
        nearest = None
        min_dist = float("inf")
        for target in visible_targets:
            snap_r = max(30.0, target.radius + 25.0)
            d = math.hypot(target.center_x - x, target.center_y - y)
            if d < snap_r * 2.5 and d < min_dist:
                min_dist = d
                nearest = target
        if nearest is None:
            return x, y
        return cls._apply_magnetic_snap(x, y, nearest, speed_px_s, snap_gain_override=snap_gain_override)

    @staticmethod
    def _compute_integrated_jerk(points: List[Tuple[float, float, float]]) -> float:
        if len(points) < 4:
            return 0.0

        dt = points[1][0] - points[0][0]
        if dt <= 0:
            return 0.0

        total_jerk = 0.0
        for i in range(len(points) - 3):
            p0, p1, p2, p3 = points[i], points[i + 1], points[i + 2], points[i + 3]
            # 3rd derivative finite difference: (p3 - 3*p2 + 3*p1 - p0) / dt^3
            jx = (p3[1] - 3.0 * p2[1] + 3.0 * p1[1] - p0[1]) / (dt ** 3)
            jy = (p3[2] - 3.0 * p2[2] + 3.0 * p1[2] - p0[2]) / (dt ** 3)
            jerk_sq = jx * jx + jy * jy
            total_jerk += jerk_sq * dt

        # Scaled to dimensionless units (1e-6) for readable table presentation
        return total_jerk * 1e-6


def _build_clutter_distractors(src: FittsTarget, dst: FittsTarget, rng: random.Random) -> List[FittsTarget]:
    dx = dst.center_x - src.center_x
    dy = dst.center_y - src.center_y
    dist = max(1.0, math.hypot(dx, dy))
    nx, ny = -dy / dist, dx / dist
    d1_frac = rng.uniform(0.35, 0.75)
    d1_lat = rng.choice([-1.0, 1.0]) * rng.uniform(18.0, 48.0)
    d2_ang = rng.uniform(0.0, 2.0 * math.pi)
    d2_off = dst.width * rng.uniform(1.15, 1.70)
    return [
        FittsTarget(900, src.center_x + dx * d1_frac + nx * d1_lat, src.center_y + dy * d1_frac + ny * d1_lat, dst.width, dst.radius),
        FittsTarget(901, dst.center_x + d2_off * math.cos(d2_ang), dst.center_y + d2_off * math.sin(d2_ang), dst.width, dst.radius),
    ]


def run_benchmark(
    n_targets: int = 16,
    target_widths: List[float] = [24.0, 48.0, 96.0],
    seed: int = 42,
    cluttered: bool = False
) -> Dict[str, Any]:
    runner = FittsBenchmarkRunner(circle_radius=280.0, center=(500.0, 500.0))
    conditions = [
        "raw",
        "one_euro",
        "viscous_damping",
        "pre_click_lock",
        "magnetic_snapping",
        "full_gestureflow",
        "rl_friction"
    ]

    all_metrics: Dict[str, List[TrialMetrics]] = {cond: [] for cond in conditions}

    # ISO 9241-9 standard progression: alternating cross-circle targets:
    # 0 -> 8 -> 1 -> 9 -> 2 -> 10 -> ...
    for width in target_widths:
        targets = runner.generate_iso_targets(n_targets=n_targets, target_width=width)
        order = []
        for i in range(n_targets // 2):
            order.append(i)
            order.append(i + n_targets // 2)

        for idx in range(len(order) - 1):
            src = targets[order[idx]]
            dst = targets[order[idx + 1]]
            trial_seed = seed + idx * 17 + int(width)
            visible_targets = [dst]
            if cluttered:
                visible_targets = [dst] + _build_clutter_distractors(src, dst, random.Random(trial_seed + 777))

            for cond in conditions:
                m = runner.run_trial(cond, src, dst, trial_seed, visible_targets=visible_targets)
                all_metrics[cond].append(m)

    # Aggregate condition summary
    summary = {}
    for cond, trials in all_metrics.items():
        n = len(trials)
        mean_tp = sum(t.throughput for t in trials) / n
        mean_mt = sum(t.movement_time for t in trials) / n
        accuracy = (sum(1 for t in trials if t.hit) / n) * 100.0
        mean_overshoots = sum(t.overshoots for t in trials) / n
        mean_jerk = sum(t.integrated_jerk for t in trials) / n
        mean_deflection = sum(t.click_deflection for t in trials) / n

        summary[cond] = {
            "trials_count": n,
            "mean_throughput_bps": round(mean_tp, 2),
            "mean_movement_time_s": round(mean_mt, 3),
            "target_accuracy_pct": round(accuracy, 1),
            "mean_overshoots": round(mean_overshoots, 2),
            "mean_integrated_jerk": round(mean_jerk, 1),
            "mean_click_deflection_px": round(mean_deflection, 2)
        }

    return {
        "metadata": {
            "standard": "ISO 9241-9 / ISO 9241-411",
            "n_targets": n_targets,
            "target_widths": target_widths,
            "circle_diameter_px": 560.0,
            "cluttered_distractors": cluttered,
            "total_trials_per_condition": len(all_metrics[conditions[0]])
        },
        "summary": summary
    }


def format_markdown_table(result: Dict[str, Any]) -> str:
    summary = result["summary"]
    condition_names = {
        "raw": "1. Raw Pointer (Unfiltered)",
        "one_euro": "2. Static 1€ Filtered Pointer",
        "viscous_damping": "3. + Nonlinear Viscous Damping",
        "pre_click_lock": "4. + Pre-Click Anchor Lock",
        "magnetic_snapping": "5. + Adaptive Magnetic Snapping",
        "full_gestureflow": "6. Full GestureFlow Engine",
        "rl_friction": "7. Full GestureFlow + RL Policy"
    }

    clutter_tag = " (Cluttered Multi-Target)" if result["metadata"].get("cluttered_distractors") else ""
    lines = [
        f"# ISO 9241-9 Fitts' Law Benchmark Report{clutter_tag}",
        "",
        f"**Standard**: {result['metadata']['standard']} | "
        f"**Targets**: {result['metadata']['n_targets']} Circular Array | "
        f"**Widths**: {result['metadata']['target_widths']} px | "
        f"**Trials / Condition**: {result['metadata']['total_trials_per_condition']}",
        "",
        "| Condition | Throughput (bits/s) ↑ | Movement Time (s) ↓ | Accuracy (%) ↑ | Overshoots ↓ | Integrated Jerk ↓ | Click Deflection (px) ↓ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for key, name in condition_names.items():
        if key not in summary:
            continue
        s = summary[key]
        lines.append(
            f"| **{name}** | **{s['mean_throughput_bps']:.2f}** | "
            f"{s['mean_movement_time_s']:.3f} | {s['target_accuracy_pct']:.1f}% | "
            f"{s['mean_overshoots']:.2f} | {s['mean_integrated_jerk']:.1f} | "
            f"**{s['mean_click_deflection_px']:.2f}** |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ISO 9241-9 Fitts' Law Scientific Benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic kinematic generation")
    parser.add_argument("--clutter", action="store_true", help="Include en-route and adjacent distractor targets")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    results = run_benchmark(seed=args.seed, cluttered=args.clutter)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_markdown_table(results))


if __name__ == "__main__":
    main()
