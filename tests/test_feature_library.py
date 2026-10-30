"""Library features on synthetic shapes; needs the `breast-agent` env (scikit-image, OpenCV), no data, no Ollama.

    ~/miniforge3/envs/breast-agent/bin/python -m unittest tests.test_feature_library
"""

import unittest

import numpy as np

from agent.features import library
from agent.features.sandbox import synthetic_cases


class Library(unittest.TestCase):
    def setUp(self):
        self.cases = synthetic_cases()

    def test_every_feature_is_finite_and_tied_to_the_catalog(self):
        for name, (_, key, _) in library.REGISTRY.items():
            self.assertTrue(library._catalog_row(key), name)
            for case, (img, m) in self.cases.items():
                if name == "posterior_ratio" and case == "small_blob":
                    continue
                self.assertTrue(np.isfinite(library.compute(name, img, m)), (name, case))

    def test_star_is_less_regular_than_circle(self):
        c, s = self.cases["circle"], self.cases["star"]
        for name in ("solidity", "convexity", "circularity"):
            self.assertLess(library.compute(name, *s), library.compute(name, *c) - 0.1, name)
        self.assertGreater(library.compute("lobulations", *s), library.compute("lobulations", *c))

    def test_menu_lists_unimplemented_and_unmeasurable_rows(self):
        status = {r["status"] for r in library.menu()}
        self.assertEqual(status, {"library", "not implemented", "not measurable in our data"})

    def test_tiny_outline_is_refused(self):
        img = np.zeros((64, 64)); m = np.zeros((64, 64), bool); m[30:33, 30:33] = True
        with self.assertRaises(ValueError):
            library.compute("solidity", img, m)


if __name__ == "__main__":
    unittest.main()
