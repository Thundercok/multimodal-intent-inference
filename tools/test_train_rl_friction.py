#!/usr/bin/env python3
"""
Unit tests for Phase 3 PPO RL Dynamic Friction Trainer (tools/train_rl_friction.py).
"""

import os
import sys
import unittest
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.bench_fitts import FittsTarget
from tools.train_rl_friction import (
    ACTION_DIM,
    CUTOFF_MAX,
    CUTOFF_MIN,
    SNAP_GAIN_MAX,
    SNAP_GAIN_MIN,
    STATE_DIM,
    ActorCriticMLP,
    FittsRLEnvironment,
    actions_from_raw,
    build_state_vector,
    train_ppo,
)


class TestTrainRLFriction(unittest.TestCase):
    def test_observation_zero_oracle_geometry(self):
        # Distractor at (460, 500) is closer to cursor (430, 500) than true target (700, 500)
        distractor = FittsTarget(99, 460.0, 500.0, 48.0, 24.0)
        true_dst = FittsTarget(0, 700.0, 500.0, 48.0, 24.0)

        obs_approaching = build_state_vector(
            cursor_x=430.0,
            cursor_y=500.0,
            vel_x=300.0,
            vel_y=0.0,
            speed_px_s=300.0,
            accel_px_s2=-1500.0,
            visible_targets=[true_dst, distractor],
            aperture=0.72,
        )
        self.assertEqual(obs_approaching.shape, (STATE_DIM,))
        self.assertAlmostEqual(float(obs_approaching[0]), 0.5, places=4)
        self.assertAlmostEqual(float(obs_approaching[1]), -0.5, places=4)
        self.assertEqual(float(obs_approaching[2]), 1.0)
        self.assertAlmostEqual(float(obs_approaching[4]), 1.0, places=4)  # approaching nearest

        # Once cursor passes distractor at (480, 500), radial alignment relative to nearest distractor turns -1
        obs_receding = build_state_vector(
            cursor_x=480.0,
            cursor_y=500.0,
            vel_x=500.0,
            vel_y=0.0,
            speed_px_s=500.0,
            accel_px_s2=0.0,
            visible_targets=[true_dst, distractor],
            aperture=0.72,
        )
        self.assertAlmostEqual(float(obs_receding[4]), -1.0, places=4)

    def test_action_bounds_mapping(self):
        c_low, g_low = actions_from_raw(np.array([-1.0, -1.0]))
        c_high, g_high = actions_from_raw(np.array([1.0, 1.0]))
        self.assertAlmostEqual(c_low, CUTOFF_MIN, places=5)
        self.assertAlmostEqual(g_low, SNAP_GAIN_MIN, places=5)
        self.assertAlmostEqual(c_high, CUTOFF_MAX, places=5)
        self.assertAlmostEqual(g_high, SNAP_GAIN_MAX, places=5)

    def test_fast_ppo_training_smoke_and_determinism(self):
        agent, history = train_ppo(epochs=2, batch_episodes=6, seed=7)
        self.assertEqual(len(history["epoch"]), 2)
        self.assertGreater(history["val_accuracy"][-1], 80.0)

        env = FittsRLEnvironment()
        obs = env.reset(seed=2500, cluttered=True)
        cutoff, gain = agent.predict_deterministic(obs)
        self.assertGreaterEqual(cutoff, CUTOFF_MIN)
        self.assertLessEqual(cutoff, CUTOFF_MAX)
        self.assertGreaterEqual(gain, SNAP_GAIN_MIN)
        self.assertLessEqual(gain, SNAP_GAIN_MAX)


if __name__ == "__main__":
    unittest.main()
