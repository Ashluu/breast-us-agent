"""Feature library: library-function implementations the agent calls before it writes any code (advisor, 30 Oct 2026).

Every feature is f(image, mask) -> float on the grayscale B-mode image (2-D, 0-255) and the U-Net lesion
outline (2-D bool). Implementations follow scripts/feature_library_survey.py (perimeters with
regionprops.perimeter_crofton; cv2.arcLength overcounts pixel staircases). Each entry points to its row in
agent/features/feature_catalog.csv (built from the open-access corpus, data/knowledge/), which gives the definition and the suspicious direction from the
literature; the LLM sees those, not our own guesses.

PyRadiomics 2-D features (shape2D, firstorder, glcm, glrlm, glszm, gldm, ngtdm; binWidth fixed at 25) are
reachable as "radiomics:<class>_<Name>", e.g. "radiomics:glcm_Contrast". Needs the `breast-agent` env.

    from agent.features import library
    library.compute("solidity", image, mask)
    library.menu()            # what the LLM is shown: implemented features + catalog rows with no implementation
"""
from __future__ import annotations

import functools
import logging
import os

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature as skf
from skimage import filters, measure, morphology

CATALOG = os.path.join(os.path.dirname(__file__), "feature_catalog.csv")   # copy of data/knowledge/feature_catalog_draft.csv
RADIOMICS_BIN_WIDTH = 25
RADIOMICS_CLASSES = ("shape2D", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")


def _props(mask):
    return measure.regionprops(mask.astype(np.uint8))[0]


def _contour(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea)


# ---- shape (outline only) ----------------------------------------------------------------------------------

def solidity(image, mask):
    return float(_props(mask).solidity)


def convexity(image, mask):
    return float(cv2.arcLength(cv2.convexHull(_contour(mask)), True) / _props(mask).perimeter_crofton)


def circularity(image, mask):
    p = _props(mask)
    return float(4 * np.pi * p.area / p.perimeter_crofton ** 2)


def eccentricity(image, mask):
    return float(_props(mask).eccentricity)


def aspect_ratio(image, mask):
    p = _props(mask)
    return float(p.axis_major_length / max(p.axis_minor_length, 1e-6))


def depth_to_width(image, mask):
    r0, c0, r1, c1 = _props(mask).bbox
    return float((r1 - r0) / (c1 - c0))


def extent(image, mask):
    return float(_props(mask).extent)


def enc(image, mask):
    (_, _), (d1, d2), _ = cv2.fitEllipse(_contour(mask))
    a, b = d1 / 2, d2 / 2
    return float(_props(mask).perimeter_crofton / (np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))))


def lobulations(image, mask, min_depth_frac=0.05):
    c = _contour(mask)
    d = cv2.convexityDefects(c, cv2.convexHull(c, returnPoints=False))
    if d is None:
        return 0.0
    depth = d.reshape(-1, 4)[:, 3] / 256.0                      # (N,1,4) in OpenCV 4, (N,4) in OpenCV 5
    return float((depth > min_depth_frac * _props(mask).equivalent_diameter_area).sum())


def radial_roughness(image, mask):
    c = _contour(mask)[:, 0, :].astype(float)
    cy, cx = _props(mask).centroid
    r = np.hypot(c[:, 0] - cx, c[:, 1] - cy)
    return float(r.std() / r.mean())


def fractal_dimension(image, mask):
    edge = mask & ~ndi.binary_erosion(mask)
    sizes = 2 ** np.arange(1, int(np.log2(min(mask.shape))) - 1)
    counts = [np.add.reduceat(np.add.reduceat(edge, np.arange(0, edge.shape[0], s), 0),
                              np.arange(0, edge.shape[1], s), 1).astype(bool).sum() for s in sizes]
    return float(-np.polyfit(np.log(sizes), np.log(counts), 1)[0])


# ---- intensity (image + outline) ---------------------------------------------------------------------------

def margin_sharpness(image, mask, band=3):
    g = filters.sobel(image.astype(float) / 255.0)
    ring = (ndi.distance_transform_edt(mask) <= band) & (ndi.distance_transform_edt(~mask) <= band)
    return float(g[ring].mean())


def edge_contrast(image, mask, band=5):
    inner = mask & (ndi.distance_transform_edt(mask) <= band)
    outer = ~mask & (ndi.distance_transform_edt(~mask) <= band)
    return float(image[outer].mean() - image[inner].mean())


def posterior_ratio(image, mask):
    r0, c0, r1, c1 = _props(mask).bbox
    h, w = r1 - r0, c1 - c0
    b0, b1 = r1, min(image.shape[0], r1 + max(h // 2, 1))
    post = image[b0:b1, c0:c1]
    lat = np.concatenate([image[b0:b1, max(0, c0 - w):c0].ravel(), image[b0:b1, c1:min(image.shape[1], c1 + w)].ravel()])
    if post.size == 0 or lat.size == 0:
        return float("nan")                                      # lesion touches the bottom or both sides
    return float(post.mean() / max(lat.mean(), 1e-6))


def echogenicity_ratio(image, mask):
    ring = ndi.binary_dilation(mask, morphology.disk(10)) & ~mask
    return float(image[mask].mean() / max(image[ring].mean(), 1e-6))


def echo_entropy(image, mask):
    return float(measure.shannon_entropy(image[mask].astype(np.uint8)))


def glcm_contrast(image, mask, levels=32):
    q = (image.astype(float) / 256 * (levels - 1)).astype(np.uint8) + 1
    q[~mask] = 0
    m = skf.graycomatrix(q, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=levels + 1, symmetric=True)
    m = m[1:, 1:].astype(float)
    m /= np.maximum(m.sum(axis=(0, 1), keepdims=True), 1)
    return float(skf.graycoprops(m, "contrast").mean())


# ---- registry: name -> (function, catalog row in feature_catalog_draft.csv, one-line note) ------------------
REGISTRY = {
    "solidity": (solidity, "solidity", "lesion area / convex hull area; 1 = fully convex"),
    "convexity": (convexity, "convexity", "convex hull perimeter / lesion perimeter; 1 = smooth"),
    "circularity": (circularity, "circularity (2D)", "4*pi*area / perimeter^2; 1 = circle"),
    "eccentricity": (eccentricity, "eccentricity", "0 = circle, towards 1 = elongated"),
    "aspect_ratio": (aspect_ratio, "long axis-to-short axis ratio (L:S)", "major / minor axis of the fitted ellipse"),
    "depth_to_width": (depth_to_width, "depth-to-width ratio / aspect ratio", "height / width in image axes; > 1 = taller than wide"),
    "extent": (extent, "extent (rectangularity)", "lesion area / bounding box area"),
    "enc": (enc, "elliptic-normalized circumference (ENC) / elliptic compactness", "perimeter / perimeter of the fitted ellipse; 1 = ellipse"),
    "lobulations": (lobulations, "number of contour peaks (NumPeaks) / lobulation count", "convexity defects deeper than 5% of the equivalent diameter"),
    "radial_roughness": (radial_roughness, "normalized radial length (NRL) mean / SD / entropy", "SD / mean of centre-to-border distance"),
    "fractal_dimension": (fractal_dimension, "fractal dimension of contour (shape complexity)", "box-counting dimension of the border; 1 = smooth line"),
    "margin_sharpness": (margin_sharpness, "edge indistinctness / margin sharpness (gradient across boundary)", "mean Sobel gradient in a 3-px band around the border"),
    "edge_contrast": (edge_contrast, "echogenicity (lesion vs surrounding mean intensity)", "mean of a 5-px ring outside minus a 5-px ring inside"),
    "posterior_ratio": (posterior_ratio, "posterior acoustic difference (Shadow = mean I_posterior - mean I_tumo", "mean below the lesion / mean beside it at the same depth; < 1 shadowing, > 1 enhancement"),
    "echogenicity_ratio": (echogenicity_ratio, "echogenicity (lesion vs surrounding mean intensity)", "mean inside / mean of a 10-px surrounding ring; < 1 hypoechoic"),
    "echo_entropy": (echo_entropy, "internal heterogeneity entropy (neighbour-difference entropy)", "Shannon entropy of the intensities inside the lesion"),
    "glcm_contrast": (glcm_contrast, "GLCM features (contrast, correlation, energy, homogeneity, entropy, cl", "GLCM contrast inside the lesion, 32 levels"),
}


@functools.lru_cache(maxsize=1)
def _catalog() -> pd.DataFrame:
    return pd.read_csv(CATALOG)


def _catalog_row(key: str) -> dict:
    c = _catalog()
    hit = c[c.feature.str.startswith(key)]
    return hit.iloc[0].to_dict() if len(hit) else {}


@functools.lru_cache(maxsize=1)
def _radiomics_extractor():
    from radiomics import featureextractor
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    ext = featureextractor.RadiomicsFeatureExtractor(force2D=True, force2DDimension=0, label=1, binWidth=RADIOMICS_BIN_WIDTH)
    ext.disableAllFeatures()
    for cls in RADIOMICS_CLASSES:
        ext.enableFeatureClassByName(cls)
    return ext


def radiomics(image, mask) -> dict:
    import SimpleITK as sitk
    out = _radiomics_extractor().execute(sitk.GetImageFromArray(image[None].astype(np.float32)),
                                         sitk.GetImageFromArray(mask[None].astype(np.uint8)))
    return {k.replace("original_", ""): float(v) for k, v in out.items() if k.startswith("original_")}


def has(name: str) -> bool:
    return name in REGISTRY or name.startswith("radiomics:")


def compute(name: str, image: np.ndarray, mask: np.ndarray) -> float:
    image, mask = np.asarray(image, dtype=float), np.asarray(mask, dtype=bool)
    if mask.sum() < 20:
        raise ValueError("outline smaller than 20 pixels: no lesion to measure")
    if name.startswith("radiomics:"):
        feats = radiomics(image, mask)
        key = name.split(":", 1)[1]
        if key not in feats:
            raise KeyError(f"unknown PyRadiomics feature {key!r}")
        return feats[key]
    if name not in REGISTRY:
        raise KeyError(f"{name!r} is not in the library")
    return REGISTRY[name][0](image, mask)


def menu() -> list[dict]:
    """What the LLM may request: library features with the literature's definition and direction, plus the
    catalog rows that have no implementation (candidates for generated code, or not measurable in our data)."""
    used = {v[1] for v in REGISTRY.values()}
    rows = []
    for name, (_, key, note) in REGISTRY.items():
        r = _catalog_row(key)
        rows.append(dict(name=name, status="library", computes=note, category=r.get("bi_rads_category"),
                         literature_direction=r.get("suspicious_direction"), sources=r.get("source_ids")))
    for _, r in _catalog().iterrows():
        if any(r.feature.startswith(k) for k in used):
            continue
        measurable = str(r.in_our_data).lower().startswith("yes")
        rows.append(dict(name=r.feature, status="not implemented" if measurable else "not measurable in our data",
                         computes=r.definition, category=r.bi_rads_category, literature_direction=r.suspicious_direction,
                         sources=r.source_ids))
    return rows
