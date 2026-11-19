#!/usr/bin/env python3
"""
Unit tests for Experiment 001 Context-Conditioned Intent Fusion (tools/eval_context_fusion.py).
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.eval_context_fusion import (
    compute_context_priors,
    fuse_probabilities,
    run_context_fusion_benchmark,
)


class TestContextFusion(unittest.TestCase):
    def test_unavailable_context_is_uniform(self):
        priors = compute_context_priors(
            app_category="unknown",
            focused_role=None,
            has_active_app=False,
            has_focused_element=False,
            has_window_state=False,
        )
        for v in priors.values():
            self.assertEqual(v, 1.0)

    def test_typing_context_suppresses_weak_pre_click_only(self):
        priors = compute_context_priors("code_editor", "AXTextArea")
        weak = {"idle": 0.25, "hover": 0.05, "pre_click": 0.68, "drag": 0.01, "scroll": 0.01}
        top_w, pre_w, fused_w = fuse_probabilities(weak, priors, is_typing_ctx=True)
        self.assertFalse(pre_w)
        self.assertEqual(top_w, "idle")
        self.assertAlmostEqual(sum(fused_w.values()), 1.0, places=5)

        strong = {"idle": 0.03, "hover": 0.01, "pre_click": 0.95, "drag": 0.005, "scroll": 0.005}
        _, pre_s, _ = fuse_probabilities(strong, priors, is_typing_ctx=True)
        self.assertTrue(pre_s)

    def test_benchmark_improves_misfire_and_scroll_metrics(self):
        res = run_context_fusion_benchmark(n_samples=200, seed=42)
        self.assertLess(
            res["typing_false_positive_rate_pct"]["context_fused"],
            res["typing_false_positive_rate_pct"]["visual_only"] * 0.25,
        )
        self.assertEqual(res["deliberate_click_recall_pct"]["context_fused"], 100.0)
        self.assertGreater(
            res["browser_scroll_disambiguation_pct"]["context_fused"],
            res["browser_scroll_disambiguation_pct"]["visual_only"],
        )


if __name__ == "__main__":
    unittest.main()
