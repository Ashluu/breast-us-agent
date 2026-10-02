"""Checks of the closed loop's logic that need no checkpoint, no data and no Ollama.

    python -m unittest tests.test_agent_loop
"""

import unittest

from agent.loop import FALLBACK_ORDER, VIEWS, _constrain, fixed_policy, run_loop
from agent.tools.imaging_tools import generate_report


def _scores(table):
    return lambda name: table[name]


class Constrain(unittest.TestCase):
    def test_rejects_unknown_scored_repeated_and_extra(self):
        accepted, rejected = _constrain(["roi", "zoom", "full", "roi", "box", "roi_sq"], ["roi", "box", "roi_sq"], 2)
        self.assertEqual(accepted, ["roi", "box"])
        self.assertEqual([r[0] for r in rejected], ["zoom", "full", "roi", "roi_sq"])


class Loop(unittest.TestCase):
    def test_stops_when_views_agree(self):
        out = run_loop(_scores({n: 0.80 for n in VIEWS}), list(VIEWS), fixed_policy)
        self.assertEqual(out["stop"], "stable")
        self.assertEqual(out["findings"]["n_views"], 2)
        self.assertTrue(out["findings"]["stable"])
        self.assertAlmostEqual(out["findings"]["prob_malignant"], 0.80, places=3)

    def test_runs_all_rounds_when_views_disagree(self):
        table = dict(zip(FALLBACK_ORDER, [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]))
        out = run_loop(_scores(table), list(VIEWS), fixed_policy, k=3, per_round=2)
        self.assertEqual(out["stop"], "max_rounds")
        self.assertEqual(out["findings"]["n_views"], 6)
        self.assertFalse(out["findings"]["stable"])
        # mean of logits of a symmetric table is 0.5: every view counts, none is selected
        self.assertAlmostEqual(out["findings"]["prob_malignant"], 0.5, places=3)

    def test_invalid_proposal_falls_back_to_fixed_order(self):
        bad = lambda state: (["not_a_view"], "llm", "{}")
        out = run_loop(_scores({n: 0.6 for n in VIEWS}), list(VIEWS), bad)
        self.assertEqual(out["rounds"][0]["source"], "fallback")
        self.assertEqual(out["rounds"][0]["accepted"], FALLBACK_ORDER[:2])
        self.assertEqual(out["rounds"][0]["rejected"], [["not_a_view", "unknown view"]])

    def test_without_a_mask_only_mask_free_views_are_used(self):
        menu = [n for n, v in VIEWS.items() if not v.needs_mask]
        out = run_loop(_scores({"full": 0.2, "full_flip": 0.6}), menu, fixed_policy)
        self.assertEqual(set(out["views"]), {"full", "full_flip"})
        self.assertFalse(out["findings"]["uses_expert_mask"])
        self.assertEqual(out["stop"], "menu_exhausted")

    def test_policy_never_sees_a_label(self):
        seen = []
        def spy(state):
            seen.append(state)
            return fixed_policy(state)
        run_loop(_scores({n: 0.5 + 0.05 * i for i, n in enumerate(VIEWS)}), list(VIEWS), spy)
        allowed = {"round", "remaining", "scores", "per_round", "stable_spread", "aggregate", "spread"}
        self.assertTrue(all(set(s) == allowed for s in seen))


class Report(unittest.TestCase):
    def test_template_carries_the_loop_fields(self):
        text = generate_report.invoke({"findings_json": {
            "modality": "us", "prob_malignant": 0.7, "confidence": 0.4, "n_views": 4,
            "view_spread": 0.31, "stable": False, "uses_expert_mask": True}})
        self.assertIn("combines 4 views", text)
        self.assertIn("unstable", text)
        self.assertIn("expert mask", text)

    def test_template_without_loop_fields_is_unchanged(self):
        text = generate_report.invoke({"findings_json": {"modality": "us", "prob_malignant": 0.2}})
        self.assertNotIn("views", text)
        self.assertNotIn("expert mask", text)


if __name__ == "__main__":
    unittest.main()
