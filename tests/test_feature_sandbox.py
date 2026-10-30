"""Checks of the generated-feature sandbox that need no checkpoint, no data and no Ollama.

    python -m unittest tests.test_feature_sandbox
"""

import unittest

from agent.features.sandbox import static_check, test

GOOD = '''
import numpy as np
from scipy import ndimage as ndi


def feature(image, mask):
    """Mean intensity in an 8-pixel ring outside the lesion minus the mean inside."""
    ring = ndi.binary_dilation(mask, iterations=8) & ~mask
    return float(image[ring].mean() - image[mask].mean())
'''


class Static(unittest.TestCase):
    def test_accepts_a_plain_feature(self):
        self.assertTrue(static_check(GOOD).ok)

    def test_rejects_os_open_and_dunders(self):
        for bad in ("import os\ndef feature(image, mask):\n    return 0.0",
                    "def feature(image, mask):\n    return float(len(open('/etc/hosts').read()))",
                    "def feature(image, mask):\n    return image.__class__.__name__",
                    "import subprocess\ndef feature(image, mask):\n    return 1.0",
                    "x = 1\ndef feature(image, mask):\n    return x"):
            self.assertFalse(static_check(bad).ok, bad)

    def test_rejects_wrong_signature_and_name(self):
        self.assertFalse(static_check("def feature(img):\n    return 0.0").ok)
        self.assertFalse(static_check("def solidity(image, mask):\n    return 0.0").ok)


class Run(unittest.TestCase):
    def test_good_feature_passes_and_separates_shapes(self):
        c = test(GOOD)
        self.assertTrue(c.ok, c.errors)
        self.assertGreater(c.values["circle"], 40)              # lesion drawn ~80 darker than background

    def test_constant_feature_is_rejected(self):
        c = test("def feature(image, mask):\n    return 1.0")
        self.assertFalse(c.ok)
        self.assertIn("same value", c.errors[0])

    def test_crash_and_nan_are_reported(self):
        self.assertFalse(test("def feature(image, mask):\n    return 1 / 0").ok)
        self.assertFalse(test("import numpy as np\ndef feature(image, mask):\n    return float(np.nan)").ok)

    def test_infinite_loop_times_out(self):
        c = test("def feature(image, mask):\n    while True:\n        pass")
        self.assertFalse(c.ok)

    def test_solidity_written_by_hand_ranks_star_below_circle(self):
        src = '''
import numpy as np
from scipy.spatial import ConvexHull


def feature(image, mask):
    ys, xs = np.nonzero(mask)
    pts = np.concatenate([np.stack([ys + dy, xs + dx], 1) for dy in (0, 1) for dx in (0, 1)])
    return float(mask.sum() / ConvexHull(pts).volume)
'''
        c = test(src)
        self.assertTrue(c.ok, c.errors)
        self.assertLess(c.values["star"], c.values["circle"] - 0.1)


if __name__ == "__main__":
    unittest.main()
