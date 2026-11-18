#!/usr/bin/env python3
"""
Phase 3 RL Trainer: PPO Continuous Control for Dynamic Viscous Damping & Magnetic Snapping.

Trains a compact 2-layer MLP policy (6 -> 32 -> 32 -> 2, tanh) via Proximal Policy Optimization (PPO)
using pure NumPy against the Flash & Hogan minimum-jerk Fitts' Law environment with:
  - 8-12 Hz physiological tremor + sensor jitter + Heisenberg click twitch
  - Multi-target clutter (en-route & adjacent distractor UI targets)
  - ZERO oracle target leakage: observation uses ONLY nearest visible target geometry

Exports:
  1. CoreML NeuralNetwork (.mlmodel + .mlmodelc) -> GestureFlowApp/Resources/DynamicFrictionPolicy.mlmodel
  2. Portable JSON weights artifact -> GestureFlowApp/Resources/dynamic_friction_policy.json
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.bench_fitts import FittsKinematicSimulator, FittsTarget, OneEuroFilter1D

STATE_DIM = 6
ACTION_DIM = 2
HIDDEN_DIM = 32

CUTOFF_MIN = 0.25
CUTOFF_MAX = 4.00
SNAP_GAIN_MIN = 0.00
SNAP_GAIN_MAX = 0.85


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -15.0, 15.0))


def actions_from_raw(raw_tanh: np.ndarray) -> Tuple[float, float]:
    """Maps [-1, 1]^2 policy output to physical (cutoff_hz, snap_gain)."""
    u0 = float(np.clip(raw_tanh[0], -1.0, 1.0))
    u1 = float(np.clip(raw_tanh[1], -1.0, 1.0))
    cutoff_hz = CUTOFF_MIN + (CUTOFF_MAX - CUTOFF_MIN) * 0.5 * (u0 + 1.0)
    snap_gain = SNAP_GAIN_MIN + (SNAP_GAIN_MAX - SNAP_GAIN_MIN) * 0.5 * (u1 + 1.0)
    return cutoff_hz, snap_gain


def build_state_vector(
    cursor_x: float,
    cursor_y: float,
    vel_x: float,
    vel_y: float,
    speed_px_s: float,
    accel_px_s2: float,
    visible_targets: List[FittsTarget],
    aperture: float,
) -> np.ndarray:
    """
    Constructs the exact 6D target-agnostic observation vector shared with Swift:
      0: norm_speed        = min(2.0, speed / 600.0)
      1: norm_signed_accel = clip(accel / 3000.0, -2.0, 2.0)
      2: has_target_prox   = 1.0 if nearest within 2.5 * snap_radius else 0.0
      3: norm_target_dist  = min(1.0, dist / (2.5 * snap_radius))
      4: radial_alignment  = cos(theta(dir_to_nearest, velocity)) in [-1, 1]
      5: aperture_ratio    = clip(aperture, 0.0, 1.0)
    """
    norm_speed = min(2.0, max(0.0, speed_px_s / 600.0))
    norm_accel = float(np.clip(accel_px_s2 / 3000.0, -2.0, 2.0))

    nearest_target = None
    min_dist = float("inf")
    snap_radius = 49.0

    for t in visible_targets:
        d = math.hypot(t.center_x - cursor_x, t.center_y - cursor_y)
        r_snap = max(30.0, t.radius + 25.0)
        if d < r_snap * 2.5 and d < min_dist:
            min_dist = d
            nearest_target = t
            snap_radius = r_snap

    if nearest_target is not None:
        has_prox = 1.0
        norm_dist = min(1.0, min_dist / (2.5 * snap_radius))
        dx = nearest_target.center_x - cursor_x
        dy = nearest_target.center_y - cursor_y
        v_norm = math.hypot(vel_x, vel_y)
        if min_dist > 1e-3 and v_norm > 1e-3:
            radial_align = (dx * vel_x + dy * vel_y) / (min_dist * v_norm)
            radial_align = float(np.clip(radial_align, -1.0, 1.0))
        else:
            radial_align = 0.0
    else:
        has_prox = 0.0
        norm_dist = 1.0
        radial_align = 0.0

    ap_ratio = float(np.clip(aperture, 0.0, 1.0))
    return np.array(
        [norm_speed, norm_accel, has_prox, norm_dist, radial_align, ap_ratio],
        dtype=np.float32,
    )


def apply_parametric_snap(
    x: float,
    y: float,
    visible_targets: List[FittsTarget],
    snap_gain: float,
) -> Tuple[float, float]:
    """Nearest-target Gaussian potential pull parameterized by policy snap_gain."""
    if snap_gain <= 1e-4 or not visible_targets:
        return x, y

    nearest = None
    min_dist = float("inf")
    for t in visible_targets:
        snap_r = max(30.0, t.radius + 25.0)
        d = math.hypot(t.center_x - x, t.center_y - y)
        if d < snap_r * 2.5 and d < min_dist:
            min_dist = d
            nearest = t

    if nearest is None or min_dist < 1e-3:
        return x, y

    snap_r = max(30.0, nearest.radius + 25.0)
    sigma = snap_r * 0.707
    gaussian = math.exp(-(min_dist * min_dist) / (2.0 * sigma * sigma))
    pull = float(np.clip(snap_gain, 0.0, SNAP_GAIN_MAX)) * gaussian
    dx = nearest.center_x - x
    dy = nearest.center_y - y
    return x + dx * pull, y + dy * pull


def generate_clean_ideal_path(
    start: Tuple[float, float],
    target: Tuple[float, float],
    num_frames: int,
) -> List[Tuple[float, float]]:
    """Reference noise-free Flash & Hogan minimum-jerk path (training supervisor signal only)."""
    pts = []
    dx = target[0] - start[0]
    dy = target[1] - start[1]
    for i in range(num_frames):
        tau = i / max(1, num_frames - 1)
        s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
        pts.append((start[0] + dx * s, start[1] + dy * s))
    return pts


def sample_distractors(
    start: Tuple[float, float],
    dest: FittsTarget,
    rng: random.Random,
    n_distractors: int = 2,
) -> List[FittsTarget]:
    """Places distractor UI targets en route or flanking the true target."""
    distractors = []
    dx = dest.center_x - start[0]
    dy = dest.center_y - start[1]
    length = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux

    for idx in range(n_distractors):
        if idx == 0:
            # En-route distractor slightly off corridor axis (tests ballistic breakaway)
            frac = rng.uniform(0.35, 0.72)
            lat = rng.choice([-1.0, 1.0]) * rng.uniform(18.0, 52.0)
            cx = start[0] + dx * frac + nx * lat
            cy = start[1] + dy * frac + ny * lat
        else:
            # Adjacent neighbor distractor flanking destination (tests clutter precision)
            ang = rng.uniform(0.0, 2.0 * math.pi)
            offset = dest.width * rng.uniform(1.15, 1.75)
            cx = dest.center_x + offset * math.cos(ang)
            cy = dest.center_y + offset * math.sin(ang)

        w = dest.width
        distractors.append(
            FittsTarget(
                target_id=100 + idx,
                center_x=cx,
                center_y=cy,
                width=w,
                radius=w * 0.5,
            )
        )
    return distractors


class FittsRLEnvironment:
    """Episodic environment simulating a single Fitts' Law target acquisition trial."""

    def __init__(self, dt: float = 0.016667):
        self.dt = dt
        self.sim = FittsKinematicSimulator(dt=dt)

    def reset(self, seed: int, cluttered: bool = True) -> np.ndarray:
        rng = random.Random(seed)
        self.seed = seed
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = rng.uniform(200.0, 320.0)
        cx, cy = 500.0, 500.0
        start = (cx - radius * math.cos(angle), cy - radius * math.sin(angle))
        dest_pt = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        width = rng.choice([24.0, 36.0, 48.0, 72.0, 96.0])

        self.dest = FittsTarget(
            target_id=0,
            center_x=dest_pt[0],
            center_y=dest_pt[1],
            width=width,
            radius=width * 0.5,
        )
        self.distractors = sample_distractors(start, self.dest, rng, n_distractors=2) if cluttered else []
        self.visible_targets = [self.dest] + self.distractors

        self.frames = self.sim.generate_movement(start=start, target=dest_pt, target_width=width, seed=seed)
        self.ideal_path = generate_clean_ideal_path(start, dest_pt, len(self.frames))
        self.idx = 0

        self.filter_x = OneEuroFilter1D(min_cutoff=1.0, beta=0.035)
        self.filter_y = OneEuroFilter1D(min_cutoff=1.0, beta=0.035)
        self.prev_raw = None
        self.prev_filtered = start
        self.prev_speed = 0.0
        self.prev_vel_vec = (0.0, 0.0)
        self.prev_accel = 0.0
        self.locked_point = None
        self.was_inside = False
        self.overshoots = 0
        self.processed_pts: List[Tuple[float, float, float]] = []

        return self._observe(start[0], start[1], 0.0, 0.0, 0.0, 0.0, self.frames[0]["aperture"])

    def _observe(
        self,
        cx: float,
        cy: float,
        vx: float,
        vy: float,
        speed: float,
        accel: float,
        aperture: float,
    ) -> np.ndarray:
        return build_state_vector(cx, cy, vx, vy, speed, accel, self.visible_targets, aperture)

    def step(self, raw_action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        cutoff_hz, snap_gain = actions_from_raw(raw_action)
        fr = self.frames[self.idx]
        t, rx, ry, aperture = fr["time"], fr["x"], fr["y"], fr["aperture"]

        if self.prev_raw is not None:
            vx_raw = (rx - self.prev_raw[0]) / self.dt
            vy_raw = (ry - self.prev_raw[1]) / self.dt
            inst_speed = math.hypot(vx_raw, vy_raw)
            new_speed = 0.35 * inst_speed + 0.65 * self.prev_speed
            accel = (new_speed - self.prev_speed) / self.dt
            self.prev_vel_vec = (
                0.35 * vx_raw + 0.65 * self.prev_vel_vec[0],
                0.35 * vy_raw + 0.65 * self.prev_vel_vec[1],
            )
            self.prev_speed = new_speed
            self.prev_accel = 0.4 * accel + 0.6 * self.prev_accel
        self.prev_raw = (rx, ry)

        fx = self.filter_x.filter(rx, t, dynamic_min_cutoff=cutoff_hz)
        fy = self.filter_y.filter(ry, t, dynamic_min_cutoff=cutoff_hz)

        # Snap unfiltered tremor-damped cursor toward nearest target; freeze snapped pos on pre-click
        if self.locked_point is not None:
            px, py = self.locked_point
        else:
            px, py = apply_parametric_snap(fx, fy, self.visible_targets, snap_gain)
            if aperture <= 0.48:
                self.locked_point = (px, py)

        self.processed_pts.append((t, px, py))

        # Overshoot count
        dist_dest = math.hypot(px - self.dest.center_x, py - self.dest.center_y)
        inside = dist_dest <= self.dest.radius
        if self.was_inside and not inside:
            self.overshoots += 1
            step_overshoot = 1.0
        else:
            step_overshoot = 0.0
        self.was_inside = inside

        # Step smoothness & tracking reward (balanced magnitudes vs terminal hit + throughput)
        ix, iy = self.ideal_path[self.idx]
        tracking_err = math.hypot(px - ix, py - iy)

        jerk_step = 0.0
        if len(self.processed_pts) >= 4:
            p0, p1, p2, p3 = self.processed_pts[-4:]
            jx = (p3[1] - 3.0 * p2[1] + 3.0 * p1[1] - p0[1]) / (self.dt ** 3)
            jy = (p3[2] - 3.0 * p2[2] + 3.0 * p1[2] - p0[2]) / (self.dt ** 3)
            jerk_step = (jx * jx + jy * jy) * self.dt * 1e-6

        # Smoothness + path fidelity (total trajectory penalty ~0.5 - 1.5, strictly secondary to target acquisition)
        reward = -0.0015 * tracking_err - 1.2e-6 * jerk_step - 2.0 * step_overshoot

        self.prev_filtered = (px, py)
        self.idx += 1
        done = self.idx >= len(self.frames)

        info: Dict[str, float] = {}
        if done:
            hit = 1.0 if dist_dest <= self.dest.radius else 0.0
            shannon_id = math.log2(
                math.hypot(self.dest.center_x - self.ideal_path[0][0], self.dest.center_y - self.ideal_path[0][1])
                / self.dest.width
                + 1.0
            )
            first_entry_t = None
            for pt_t, pt_x, pt_y in self.processed_pts:
                if math.hypot(pt_x - self.dest.center_x, pt_y - self.dest.center_y) <= self.dest.radius:
                    first_entry_t = pt_t
                    break
            final_t = self.processed_pts[-1][0]
            mt = max(0.20, min(final_t, first_entry_t + 0.045)) if (first_entry_t is not None and hit > 0.5) else final_t
            tp = shannon_id / max(0.01, mt)

            # Terminal objective: Target Acquisition Accuracy (+20) + Throughput - miss distance
            reward += 20.0 * hit + 1.0 * tp - 0.25 * max(0.0, dist_dest - self.dest.radius)
            total_jerk = sum(
                (
                    ((self.processed_pts[k + 3][1] - 3 * self.processed_pts[k + 2][1] + 3 * self.processed_pts[k + 1][1] - self.processed_pts[k][1]) / (self.dt ** 3)) ** 2
                    + ((self.processed_pts[k + 3][2] - 3 * self.processed_pts[k + 2][2] + 3 * self.processed_pts[k + 1][2] - self.processed_pts[k][2]) / (self.dt ** 3)) ** 2
                )
                * self.dt
                * 1e-6
                for k in range(max(0, len(self.processed_pts) - 3))
            )
            info = {"hit": hit, "throughput": tp, "jerk": total_jerk, "overshoots": float(self.overshoots)}
            next_obs = np.zeros(STATE_DIM, dtype=np.float32)
        else:
            next_fr = self.frames[self.idx]
            next_obs = self._observe(
                px,
                py,
                self.prev_vel_vec[0],
                self.prev_vel_vec[1],
                self.prev_speed,
                self.prev_accel,
                next_fr["aperture"],
            )

        return next_obs, float(reward), done, info


class ActorCriticMLP:
    """Pure-NumPy 2-hidden-layer MLP Actor & Critic trained with PPO + Adam."""

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)

        def init_layer(fan_in: int, fan_out: int, scale: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
            w = rng.normal(0.0, scale / math.sqrt(fan_in), size=(fan_in, fan_out)).astype(np.float64)
            b = np.zeros(fan_out, dtype=np.float64)
            return w, b

        # Actor weights: 6 -> 32 -> 32 -> 2
        self.a_w0, self.a_b0 = init_layer(STATE_DIM, HIDDEN_DIM, 1.0)
        self.a_w1, self.a_b1 = init_layer(HIDDEN_DIM, HIDDEN_DIM, 1.0)
        self.a_w2, self.a_b2 = init_layer(HIDDEN_DIM, ACTION_DIM, 0.1)
        self.log_std = np.array([-0.5, -0.5], dtype=np.float64)

        # Critic weights: 6 -> 32 -> 32 -> 1
        self.c_w0, self.c_b0 = init_layer(STATE_DIM, HIDDEN_DIM, 1.0)
        self.c_w1, self.c_b1 = init_layer(HIDDEN_DIM, HIDDEN_DIM, 1.0)
        self.c_w2, self.c_b2 = init_layer(HIDDEN_DIM, 1, 0.1)

        self.adam_m: Dict[str, np.ndarray] = {}
        self.adam_v: Dict[str, np.ndarray] = {}
        self.adam_t = 0

    def actor_forward(self, s: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        z0 = s @ self.a_w0 + self.a_b0
        h0 = np.tanh(z0)
        z1 = h0 @ self.a_w1 + self.a_b1
        h1 = np.tanh(z1)
        out = np.tanh(h1 @ self.a_w2 + self.a_b2)
        return out, (s, h0, h1)

    def critic_forward(self, s: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        z0 = s @ self.c_w0 + self.c_b0
        h0 = np.tanh(z0)
        z1 = h0 @ self.c_w1 + self.c_b1
        h1 = np.tanh(z1)
        val = h1 @ self.c_w2 + self.c_b2
        return val.squeeze(-1), (s, h0, h1)

    def predict_deterministic(self, state: np.ndarray) -> Tuple[float, float]:
        s = state.astype(np.float64).reshape(1, -1)
        raw, _ = self.actor_forward(s)
        return actions_from_raw(raw[0])

    def _adam_step(self, grads: Dict[str, np.ndarray], lr: float = 3e-4):
        self.adam_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name, grad in grads.items():
            param = getattr(self, name)
            grad_clipped = np.clip(grad, -5.0, 5.0)
            if name not in self.adam_m:
                self.adam_m[name] = np.zeros_like(param)
                self.adam_v[name] = np.zeros_like(param)
            self.adam_m[name] = b1 * self.adam_m[name] + (1.0 - b1) * grad_clipped
            self.adam_v[name] = b2 * self.adam_v[name] + (1.0 - b2) * (grad_clipped * grad_clipped)
            m_hat = self.adam_m[name] / (1.0 - b1 ** self.adam_t)
            v_hat = self.adam_v[name] / (1.0 - b2 ** self.adam_t)
            setattr(self, name, param - lr * m_hat / (np.sqrt(v_hat) + eps))
        self.log_std = np.clip(self.log_std, -1.8, 0.0)

    def get_param_dict(self) -> Dict[str, np.ndarray]:
        return {
            "a_w0": self.a_w0.copy(),
            "a_b0": self.a_b0.copy(),
            "a_w1": self.a_w1.copy(),
            "a_b1": self.a_b1.copy(),
            "a_w2": self.a_w2.copy(),
            "a_b2": self.a_b2.copy(),
        }

    def load_param_dict(self, params: Dict[str, np.ndarray]):
        for k, v in params.items():
            setattr(self, k, v.copy())


def gaussian_log_prob(mu: np.ndarray, log_std: np.ndarray, act: np.ndarray) -> np.ndarray:
    std = np.exp(log_std)
    var = std * std
    return np.sum(-0.5 * (((act - mu) ** 2) / var + 2.0 * log_std + math.log(2.0 * math.pi)), axis=-1)


def heuristic_raw_action(speed_px_s: float) -> np.ndarray:
    """Analytical quadratic viscous damping + Hermite breakaway baseline in [-1, 1]^2."""
    ratio = min(1.0, max(0.0, speed_px_s / 450.0))
    cutoff = 0.40 + (2.50 - 0.40) * (ratio * ratio)
    if speed_px_s <= 200.0:
        wv = 1.0
    elif speed_px_s >= 650.0:
        wv = 0.0
    else:
        t = (speed_px_s - 200.0) / 450.0
        wv = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
    snap_gain = 0.65 * wv
    u0 = 2.0 * (cutoff - CUTOFF_MIN) / (CUTOFF_MAX - CUTOFF_MIN) - 1.0
    u1 = 2.0 * (snap_gain - SNAP_GAIN_MIN) / (SNAP_GAIN_MAX - SNAP_GAIN_MIN) - 1.0
    return np.array([u0, u1], dtype=np.float64)


def warm_start_from_heuristic(agent: ActorCriticMLP, env: FittsRLEnvironment):
    """Behavior-clones the heuristic controller as the clean initial policy for PPO fine-tuning."""
    S_bc, A_bc = [], []
    for seed in range(1000, 1080):
        obs = env.reset(seed=seed, cluttered=(seed % 2 == 0))
        done = False
        while not done:
            act = heuristic_raw_action(env.prev_speed)
            S_bc.append(obs.astype(np.float64))
            A_bc.append(act)
            obs, _, done, _ = env.step(act)

    S = np.array(S_bc, dtype=np.float64)
    Y = np.clip(np.array(A_bc, dtype=np.float64), -0.98, 0.98)
    n = len(S)

    for _ in range(120):
        mu, (_, ah0, ah1) = agent.actor_forward(S)
        d_loss_d_mu = 2.0 * (mu - Y) / n
        dz2 = d_loss_d_mu * (1.0 - mu * mu)
        da_w2 = ah1.T @ dz2
        da_b2 = np.sum(dz2, axis=0)
        dh1 = dz2 @ agent.a_w2.T
        dz1 = dh1 * (1.0 - ah1 * ah1)
        da_w1 = ah0.T @ dz1
        da_b1 = np.sum(dz1, axis=0)
        dh0 = dz1 @ agent.a_w1.T
        dz0 = dh0 * (1.0 - ah0 * ah0)
        da_w0 = S.T @ dz0
        da_b0 = np.sum(dz0, axis=0)
        agent._adam_step(
            {
                "a_w0": da_w0,
                "a_b0": da_b0,
                "a_w1": da_w1,
                "a_b1": da_b1,
                "a_w2": da_w2,
                "a_b2": da_b2,
            },
            lr=3e-3,
        )

    # Reset Adam moments and lower exploration noise for stable PPO refinement
    agent.adam_m.clear()
    agent.adam_v.clear()
    agent.adam_t = 0
    agent.log_std = np.array([-1.05, -1.05], dtype=np.float64)


def train_ppo(epochs: int = 35, batch_episodes: int = 24, seed: int = 42) -> Tuple[ActorCriticMLP, Dict[str, List[float]]]:
    rng = np.random.default_rng(seed)
    env = FittsRLEnvironment()
    agent = ActorCriticMLP(seed=seed)

    warm_start_from_heuristic(agent, env)

    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_reward": [],
        "val_throughput": [],
        "val_jerk": [],
        "val_accuracy": [],
    }
    best_score = -float("inf")
    best_params = agent.get_param_dict()

    for epoch in range(epochs):
        obs_buf, act_buf, logp_buf, adv_buf, ret_buf = [], [], [], [], []
        ep_rewards = []

        # Collect trajectories on training seeds 1000..1999 (disjoint from evaluation)
        for ep in range(batch_episodes):
            ep_seed = 1000 + (epoch * batch_episodes + ep) % 900
            cluttered = (ep % 3 != 0)
            obs = env.reset(seed=ep_seed, cluttered=cluttered)
            ep_obs, ep_act, ep_logp, ep_rew, ep_val = [], [], [], [], []

            done = False
            while not done:
                s_in = obs.astype(np.float64).reshape(1, -1)
                mu, _ = agent.actor_forward(s_in)
                val, _ = agent.critic_forward(s_in)
                std = np.exp(agent.log_std)
                a_sample = mu[0] + std * rng.normal(size=ACTION_DIM)
                a_exec = np.clip(a_sample, -1.0, 1.0)
                lp = gaussian_log_prob(mu[0], agent.log_std, a_sample)

                next_obs, rew, done, _ = env.step(a_exec)
                ep_obs.append(obs.astype(np.float64))
                ep_act.append(a_sample)
                ep_logp.append(float(lp))
                ep_rew.append(rew)
                ep_val.append(float(val[0]))
                obs = next_obs

            ep_rewards.append(sum(ep_rew))
            # Compute GAE
            T = len(ep_rew)
            advantages = np.zeros(T, dtype=np.float64)
            last_gae = 0.0
            for t in reversed(range(T)):
                next_v = 0.0 if t == T - 1 else ep_val[t + 1]
                delta = ep_rew[t] + gamma * next_v - ep_val[t]
                last_gae = delta + gamma * lam * last_gae
                advantages[t] = last_gae
            returns = advantages + np.array(ep_val, dtype=np.float64)

            obs_buf.extend(ep_obs)
            act_buf.extend(ep_act)
            logp_buf.extend(ep_logp)
            adv_buf.extend(advantages)
            ret_buf.extend(returns)

        S = np.array(obs_buf, dtype=np.float64)
        A = np.array(act_buf, dtype=np.float64)
        OLD_LP = np.array(logp_buf, dtype=np.float64)
        ADV = np.array(adv_buf, dtype=np.float64)
        RET = np.array(ret_buf, dtype=np.float64)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        n_samples = len(S)
        # Multiple PPO optimization passes over rollout buffer
        for _ in range(4):
            perm = rng.permutation(n_samples)
            mb_size = min(256, n_samples)
            for start in range(0, n_samples, mb_size):
                idx = perm[start : start + mb_size]
                s_b, a_b, old_lp_b, adv_b, ret_b = S[idx], A[idx], OLD_LP[idx], ADV[idx], RET[idx]
                bs = len(idx)

                # --- Actor forward & PPO clipped gradient ---
                mu_b, (_, ah0, ah1) = agent.actor_forward(s_b)
                std_b = np.exp(agent.log_std)
                var_b = std_b * std_b
                lp_b = gaussian_log_prob(mu_b, agent.log_std, a_b)
                ratio = np.exp(np.clip(lp_b - old_lp_b, -10.0, 10.0))

                unclipped = ratio * adv_b
                clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
                use_unclipped = ((adv_b >= 0) & (ratio <= 1.0 + clip_eps)) | ((adv_b < 0) & (ratio >= 1.0 - clip_eps))

                # d(-mean(min(unclipped, clipped)))/d(lp_b)
                d_loss_d_lp = np.where(use_unclipped, -(ratio * adv_b) / bs, 0.0)
                # d(lp)/d(mu) = (a - mu) / var
                d_loss_d_mu = d_loss_d_lp[:, None] * (a_b - mu_b) / var_b
                # d(lp)/d(log_std) = ((a - mu)^2 / var - 1) + small entropy bonus
                d_loss_d_logstd = np.sum(d_loss_d_lp[:, None] * (((a_b - mu_b) ** 2) / var_b - 1.0), axis=0) - 0.005

                # Backprop through actor tanh output and hidden layers
                dz2 = d_loss_d_mu * (1.0 - mu_b * mu_b)
                da_w2 = ah1.T @ dz2
                da_b2 = np.sum(dz2, axis=0)
                dh1 = dz2 @ agent.a_w2.T
                dz1 = dh1 * (1.0 - ah1 * ah1)
                da_w1 = ah0.T @ dz1
                da_b1 = np.sum(dz1, axis=0)
                dh0 = dz1 @ agent.a_w1.T
                dz0 = dh0 * (1.0 - ah0 * ah0)
                da_w0 = s_b.T @ dz0
                da_b0 = np.sum(dz0, axis=0)

                # --- Critic MSE gradient ---
                val_b, (_, ch0, ch1) = agent.critic_forward(s_b)
                d_loss_d_val = (val_b - ret_b)[:, None] / bs
                dc_w2 = ch1.T @ d_loss_d_val
                dc_b2 = np.sum(d_loss_d_val, axis=0)
                dch1 = d_loss_d_val @ agent.c_w2.T
                dcz1 = dch1 * (1.0 - ch1 * ch1)
                dc_w1 = ch0.T @ dcz1
                dc_b1 = np.sum(dcz1, axis=0)
                dch0 = dcz1 @ agent.c_w1.T
                dcz0 = dch0 * (1.0 - ch0 * ch0)
                dc_w0 = s_b.T @ dcz0
                dc_b0 = np.sum(dcz0, axis=0)

                agent._adam_step(
                    {
                        "a_w0": da_w0,
                        "a_b0": da_b0,
                        "a_w1": da_w1,
                        "a_b1": da_b1,
                        "a_w2": da_w2,
                        "a_b2": da_b2,
                        "log_std": d_loss_d_logstd,
                        "c_w0": dc_w0,
                        "c_b0": dc_b0,
                        "c_w1": dc_w1,
                        "c_b1": dc_b1,
                        "c_w2": dc_w2,
                        "c_b2": dc_b2,
                    },
                    lr=2e-4,
                )

        # Evaluate on unseen validation seeds 2000..2029 (cluttered)
        val_hits, val_tps, val_jerks, val_overs = [], [], [], []
        for vs in range(2000, 2030):
            obs = env.reset(seed=vs, cluttered=True)
            done = False
            while not done:
                s_in = obs.astype(np.float64).reshape(1, -1)
                mu, _ = agent.actor_forward(s_in)
                obs, _, done, info = env.step(mu[0])
            val_hits.append(info["hit"])
            val_tps.append(info["throughput"])
            val_jerks.append(info["jerk"])
            val_overs.append(info["overshoots"])

        mean_tp = float(np.mean(val_tps))
        mean_jerk = float(np.mean(val_jerks))
        acc = float(np.mean(val_hits)) * 100.0
        mean_rew = float(np.mean(ep_rewards))

        # Validation selection objective: strict priority on accuracy & throughput, refined by smoothness
        val_score = acc * 2.5 + mean_tp * 3.0 - mean_jerk * 0.000008 - float(np.mean(val_overs)) * 5.0
        if val_score > best_score:
            best_score = val_score
            best_params = agent.get_param_dict()

        history["epoch"].append(epoch + 1)
        history["train_reward"].append(round(mean_rew, 3))
        history["val_throughput"].append(round(mean_tp, 2))
        history["val_jerk"].append(round(mean_jerk, 1))
        history["val_accuracy"].append(round(acc, 1))

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(
                f"[PPO Epoch {epoch + 1:2d}/{epochs}] "
                f"TrainRew={mean_rew:6.2f} | Val Acc={acc:5.1f}% | Val TP={mean_tp:4.2f} bps | Val Jerk={mean_jerk:9.1f}"
            )

    agent.load_param_dict(best_params)
    return agent, history


def export_policy_artifacts(
    agent: ActorCriticMLP,
    json_out: str,
    mlmodel_out: str,
) -> bool:
    """Saves lightweight JSON weights + CoreML .mlmodel (.mlmodelc compiled)."""
    os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
    payload = {
        "metadata": {
            "state_dim": STATE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "action_dim": ACTION_DIM,
            "cutoff_range": [CUTOFF_MIN, CUTOFF_MAX],
            "snap_gain_range": [SNAP_GAIN_MIN, SNAP_GAIN_MAX],
        },
        "weights": {
            "w0": agent.a_w0.tolist(),
            "b0": agent.a_b0.tolist(),
            "w1": agent.a_w1.tolist(),
            "b1": agent.a_b1.tolist(),
            "w2": agent.a_w2.tolist(),
            "b2": agent.a_b2.tolist(),
        },
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[Policy] Saved JSON policy weights to {json_out}")

    try:
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb

        w0 = agent.a_w0.T.astype(np.float32)
        b0 = agent.a_b0.astype(np.float32)
        w1 = agent.a_w1.T.astype(np.float32)
        b1 = agent.a_b1.astype(np.float32)
        w2 = agent.a_w2.T.astype(np.float32)
        b2 = agent.a_b2.astype(np.float32)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, STATE_DIM), dtype=ct.converters.mil.mil.types.fp32)])
        def prog(obs_features):
            x = mb.linear(x=obs_features, weight=w0, bias=b0)
            x = mb.tanh(x=x)
            x = mb.linear(x=x, weight=w1, bias=b1)
            x = mb.tanh(x=x)
            x = mb.linear(x=x, weight=w2, bias=b2)
            out = mb.tanh(x=x, name="action_tanh")
            return out

        mlmodel = ct.convert(prog, convert_to="neuralnetwork")
        os.makedirs(os.path.dirname(os.path.abspath(mlmodel_out)), exist_ok=True)
        mlmodel.save(mlmodel_out)
        print(f"[CoreML] Exported RL policy model to {mlmodel_out}")

        try:
            out_dir = os.path.dirname(os.path.abspath(mlmodel_out))
            subprocess.run(["xcrun", "coremlcompiler", "compile", mlmodel_out, out_dir], check=True, capture_output=True)
            print(f"[CoreML] Compiled .mlmodelc in {out_dir}")
        except Exception as e:
            print(f"[CoreML] xcrun coremlcompiler note: {e}")
        return True
    except ImportError as e:
        print(f"[CoreML] coremltools unavailable: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train PPO RL Dynamic Friction Policy for Phase 3")
    parser.add_argument("--epochs", type=int, default=35, help="PPO training epochs")
    parser.add_argument("--episodes", type=int, default=24, help="Episodes per PPO epoch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--json-out", default="GestureFlowApp/Resources/dynamic_friction_policy.json")
    parser.add_argument("--model-out", default="GestureFlowApp/Resources/DynamicFrictionPolicy.mlmodel")
    args = parser.parse_args()

    print("=================================================================")
    print("   GESTUREFLOW PHASE 3: PPO RL DYNAMIC FRICTION POLICY TRAINER   ")
    print("=================================================================")
    agent, _ = train_ppo(epochs=args.epochs, batch_episodes=args.episodes, seed=args.seed)
    export_policy_artifacts(agent, args.json_out, args.model_out)


if __name__ == "__main__":
    main()
