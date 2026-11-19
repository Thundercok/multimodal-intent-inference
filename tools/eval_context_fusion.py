#!/usr/bin/env python3
"""
Lightweight Evaluation Harness for Experiment 001: Context-Conditioned Multimodal Intent Fusion.

Mirrors the exact closed-form O(1) Bayesian fusion in:
  GestureFlowApp/Sources/GestureFlow/Tracking/ContextIntentFusion.swift

Evaluates Visual-Only vs. Context-Fused intent posteriors across:
  1. Typing False-Positive Suppression (borderline hand jitter in AXTextArea / AXTextField)
  2. Deliberate Pre-Click Recall Preservation (high-confidence intentional clamps punch through)
  3. Browser Scroll Disambiguation (borderline hover vs. scroll inside AXScrollArea)
"""

import argparse
import json
import random
from typing import Dict, List, Optional, Tuple

INTENT_CLASSES = ["idle", "hover", "pre_click", "drag", "scroll"]

TYPING_ROLES = {"AXTextField", "AXTextArea", "AXSecureTextField", "AXSearchField", "AXComboBox"}
SCROLL_ROLES = {"AXScrollArea", "AXWebArea", "AXTable", "AXOutline", "AXList"}
INTERACTIVE_ROLES = {"AXButton", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXSlider", "AXMenuItem", "AXTabGroup"}


def compute_context_priors(
    app_category: str = "unknown",
    focused_role: Optional[str] = None,
    window_state: str = "foreground",
    has_active_app: bool = True,
    has_focused_element: bool = True,
    has_window_state: bool = True,
) -> Dict[str, float]:
    priors = {c: 1.0 for c in INTENT_CLASSES}

    if has_active_app:
        if app_category == "browser":
            priors["scroll"] *= 1.25
            priors["pre_click"] *= 1.10
        elif app_category == "code_editor":
            priors["idle"] *= 1.15
            priors["scroll"] *= 1.15
            priors["drag"] *= 0.85
        elif app_category == "communication":
            priors["idle"] *= 1.25
            priors["pre_click"] *= 0.85
            priors["drag"] *= 0.80
        elif app_category == "media":
            priors["scroll"] *= 1.15
            priors["hover"] *= 0.90
            priors["drag"] *= 0.80
        elif app_category == "file_manager":
            priors["drag"] *= 1.30
            priors["pre_click"] *= 1.15

    if has_focused_element and focused_role:
        if focused_role in TYPING_ROLES:
            priors["idle"] *= 2.40
            priors["hover"] *= 0.85
            priors["pre_click"] *= 0.45
            priors["drag"] *= 0.55
            priors["scroll"] *= 0.65
        elif focused_role in SCROLL_ROLES:
            priors["scroll"] *= 1.35
            priors["idle"] *= 0.90
            priors["hover"] *= 1.05
            priors["pre_click"] *= 1.05
            priors["drag"] *= 0.95
        elif focused_role in INTERACTIVE_ROLES:
            priors["pre_click"] *= 1.35
            priors["drag"] *= 1.15
            priors["hover"] *= 1.10
            priors["idle"] *= 0.85
            priors["scroll"] *= 0.80

    if has_window_state and window_state == "hidden":
        priors["idle"] *= 1.50

    return priors


def fuse_probabilities(
    visual_probs: Dict[str, float],
    priors: Dict[str, float],
    is_typing_ctx: bool = False,
    pre_click_threshold: float = 0.65,
) -> Tuple[str, bool, Dict[str, float]]:
    unnorm = {c: max(0.0, visual_probs.get(c, 0.0)) * priors.get(c, 1.0) for c in INTENT_CLASSES}
    total = sum(unnorm.values())
    if total <= 1e-6:
        top_c = max(visual_probs, key=visual_probs.get)
        return top_c, (visual_probs.get("pre_click", 0.0) >= pre_click_threshold), visual_probs

    fused = {c: unnorm[c] / total for c in INTENT_CLASSES}
    top_c = max(INTENT_CLASSES, key=lambda c: fused[c])
    p_pre = fused["pre_click"]
    is_pre = (p_pre >= pre_click_threshold) or (top_c == "pre_click" and not is_typing_ctx)
    return top_c, is_pre, fused


def _sample_dirichlet_like(weights: List[float], rng: random.Random) -> Dict[str, float]:
    vals = [max(1e-4, rng.gammavariate(w, 1.0)) for w in weights]
    s = sum(vals)
    return {c: v / s for c, v in zip(INTENT_CLASSES, vals)}


def run_context_fusion_benchmark(n_samples: int = 300, seed: int = 42) -> Dict[str, Dict[str, float]]:
    rng = random.Random(seed)

    # 1. Borderline typing jitter (user fingers twitch near keyboard; visual model emits ambiguous pre_click 0.55..0.78)
    vis_typing_fp = 0
    fused_typing_fp = 0
    typing_priors = compute_context_priors("code_editor", "AXTextArea")
    for _ in range(n_samples):
        p_pre = rng.uniform(0.55, 0.78)
        p_idle = rng.uniform(0.14, 0.35)
        rem = max(0.01, 1.0 - p_pre - p_idle)
        vp = {"idle": p_idle, "hover": rem * 0.6, "pre_click": p_pre, "drag": rem * 0.2, "scroll": rem * 0.2}
        # Normalize vp
        s = sum(vp.values())
        vp = {k: v / s for k, v in vp.items()}

        if vp["pre_click"] >= 0.65 or max(vp, key=vp.get) == "pre_click":
            vis_typing_fp += 1
        _, fused_pre, _ = fuse_probabilities(vp, typing_priors, is_typing_ctx=True)
        if fused_pre:
            fused_typing_fp += 1

    # 2. Deliberate strong pinch inside code editor (user intentionally pinches strongly: p_pre in 0.90..0.98)
    vis_delib_tp = 0
    fused_delib_tp = 0
    for _ in range(n_samples):
        p_pre = rng.uniform(0.91, 0.98)
        rem = 1.0 - p_pre
        vp = {"idle": rem * 0.4, "hover": rem * 0.3, "pre_click": p_pre, "drag": rem * 0.2, "scroll": rem * 0.1}
        if vp["pre_click"] >= 0.65:
            vis_delib_tp += 1
        _, fused_pre, _ = fuse_probabilities(vp, typing_priors, is_typing_ctx=True)
        if fused_pre:
            fused_delib_tp += 1

    # 3. Borderline scroll inside Browser AXScrollArea (hover ~0.46 vs scroll ~0.41)
    vis_scroll_acc = 0
    fused_scroll_acc = 0
    browser_priors = compute_context_priors("browser", "AXScrollArea")
    for _ in range(n_samples):
        vp = _sample_dirichlet_like([1.2, 8.5, 0.5, 0.4, 8.2], rng)
        if max(vp, key=vp.get) == "scroll":
            vis_scroll_acc += 1
        top_fused, _, _ = fuse_probabilities(vp, browser_priors, is_typing_ctx=False)
        if top_fused == "scroll":
            fused_scroll_acc += 1

    return {
        "typing_false_positive_rate_pct": {
            "visual_only": round(100.0 * vis_typing_fp / n_samples, 1),
            "context_fused": round(100.0 * fused_typing_fp / n_samples, 1),
        },
        "deliberate_click_recall_pct": {
            "visual_only": round(100.0 * vis_delib_tp / n_samples, 1),
            "context_fused": round(100.0 * fused_delib_tp / n_samples, 1),
        },
        "browser_scroll_disambiguation_pct": {
            "visual_only": round(100.0 * vis_scroll_acc / n_samples, 1),
            "context_fused": round(100.0 * fused_scroll_acc / n_samples, 1),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate lightweight Bayesian Context Intent Fusion")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    res = run_context_fusion_benchmark(n_samples=args.samples, seed=args.seed)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("# Context-Conditioned Multimodal Intent Fusion Report\n")
        print("| Metric | Visual-Only | Context-Fused |")
        print("| :--- | :---: | :---: |")
        for k, v in res.items():
            print(f"| `{k}` | {v['visual_only']:.1f}% | **{v['context_fused']:.1f}%** |")


if __name__ == "__main__":
    main()
