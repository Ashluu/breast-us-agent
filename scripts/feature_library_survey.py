"""Feature library survey for the breast-US agent (B-mode image + binary lesion mask).

For each candidate BI-RADS-style feature this script gives a short reference
implementation built on library functions (scikit-image, OpenCV, SciPy,
PyRadiomics), runs it on synthetic test cases and on one real BUS-BRA case,
and prints a sanity table. Results are copied by hand into
agent/features/LIBRARY_SURVEY.md.

Run (conda env `breast-agent`, from the repo root):
    ~/miniforge3/envs/breast-agent/bin/python scripts/feature_library_survey.py
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage as ndi
from skimage import draw, feature, filters, measure, morphology

ROOT = Path(__file__).resolve().parents[1]
REAL_IMAGE = ROOT / "data/bus_bra/images/bus_0605-r.png"
REAL_ZIP = ROOT / "data/bus_bra/BUSBRA.zip"
REAL_MASK_MEMBER = "BUSBRA/Masks/mask_0605-r.png"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _props(mask):
    return measure.regionprops(mask.astype(np.uint8))[0]


def _contour(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea)


# --------------------------------------------------------------------------
# shape features (mask only)
# --------------------------------------------------------------------------
def solidity(mask):
    # skimage.measure.regionprops(...).solidity = area / convex hull area
    return float(_props(mask).solidity)


def convexity(mask):
    # convex hull perimeter (cv2.convexHull + cv2.arcLength) / regionprops.perimeter_crofton.
    # cv2.arcLength on the raw pixel contour overcounts the staircase (circle gives 0.958).
    c = _contour(mask)
    return float(cv2.arcLength(cv2.convexHull(c), True) / _props(mask).perimeter_crofton)


def circularity(mask):
    # 4*pi*A / P^2 with regionprops.area and regionprops.perimeter_crofton
    p = _props(mask)
    return float(4 * np.pi * p.area / p.perimeter_crofton ** 2)


def roundness(mask):
    # 4*A / (pi * major_axis^2), regionprops.axis_major_length
    p = _props(mask)
    return float(4 * p.area / (np.pi * p.axis_major_length ** 2))


def eccentricity(mask):
    # regionprops.eccentricity of the moment-equivalent ellipse
    return float(_props(mask).eccentricity)


def aspect_ratio(mask):
    # major / minor axis of the moment-equivalent ellipse (regionprops)
    p = _props(mask)
    return float(p.axis_major_length / max(p.axis_minor_length, 1e-6))


def taller_than_wide(mask):
    # bbox height / width in image axes (rows = depth in B-mode); >1 = taller-than-wide
    r0, c0, r1, c1 = _props(mask).bbox
    return float((r1 - r0) / (c1 - c0))


def enc(mask):
    # elliptic-normalized circumference: regionprops.perimeter_crofton / perimeter of cv2.fitEllipse
    c = _contour(mask)
    (_, _), (d1, d2), _ = cv2.fitEllipse(c)
    a, b = d1 / 2, d2 / 2
    ell_p = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))  # Ramanujan
    return float(_props(mask).perimeter_crofton / ell_p)


def lobulations(mask, min_depth_frac=0.05):
    # cv2.convexityDefects; count defects deeper than 5% of the equivalent diameter
    c = _contour(mask)
    hull = cv2.convexHull(c, returnPoints=False)
    d = cv2.convexityDefects(c, hull)
    if d is None:
        return 0
    depth_px = d.reshape(-1, 4)[:, 3] / 256.0  # (N,1,4) in OpenCV 4, (N,4) in OpenCV 5
    return int((depth_px > min_depth_frac * _props(mask).equivalent_diameter_area).sum())


def radial_roughness(mask):
    # std/mean of centroid-to-boundary distance (cv2.findContours + regionprops.centroid)
    c = _contour(mask)[:, 0, :].astype(float)
    cy, cx = _props(mask).centroid
    r = np.hypot(c[:, 0] - cx, c[:, 1] - cy)
    return float(r.std() / r.mean())


def fractal_dimension(mask):
    # box-counting dimension of the boundary (skimage.segmentation-free: mask XOR erosion)
    edge = mask & ~ndi.binary_erosion(mask)
    sizes = 2 ** np.arange(1, int(np.log2(min(mask.shape))) - 1)
    counts = [np.add.reduceat(np.add.reduceat(edge, np.arange(0, edge.shape[0], s), 0),
                              np.arange(0, edge.shape[1], s), 1).astype(bool).sum() for s in sizes]
    return float(-np.polyfit(np.log(sizes), np.log(counts), 1)[0])


# --------------------------------------------------------------------------
# intensity features (image + mask)
# --------------------------------------------------------------------------
def margin_sharpness(img, mask, band=3):
    # mean skimage.filters.sobel magnitude on a +-band px ring around the boundary,
    # normalised by the image intensity range
    g = filters.sobel(img.astype(float) / 255.0)
    dist_in = ndi.distance_transform_edt(mask)
    dist_out = ndi.distance_transform_edt(~mask)
    ring = (dist_in <= band) & (dist_out <= band)
    return float(g[ring].mean())


def edge_contrast(img, mask, band=5):
    # mean(outside ring) - mean(inside ring) using scipy distance transforms
    dist_in = ndi.distance_transform_edt(mask)
    dist_out = ndi.distance_transform_edt(~mask)
    inner = mask & (dist_in <= band)
    outer = ~mask & (dist_out <= band)
    return float(img[outer].mean() - img[inner].mean())


def posterior_acoustic_ratio(img, mask):
    # mean below the lesion (same columns, depth h/2) / mean of lateral strips at that depth
    r0, c0, r1, c1 = _props(mask).bbox
    h, w = r1 - r0, c1 - c0
    b0, b1 = r1, min(img.shape[0], r1 + max(h // 2, 1))
    post = img[b0:b1, c0:c1]
    lat = np.concatenate([img[b0:b1, max(0, c0 - w):c0].ravel(),
                          img[b0:b1, c1:min(img.shape[1], c1 + w)].ravel()])
    return float(post.mean() / max(lat.mean(), 1e-6))


def internal_echo(img, mask):
    # regionprops(intensity_image).intensity_mean, std, skimage.measure.shannon_entropy,
    # and ratio to a 10 px surrounding ring (scipy.ndimage.binary_dilation)
    vals = img[mask]
    ring = ndi.binary_dilation(mask, morphology.disk(10)) & ~mask
    return {"echo_mean": float(vals.mean()),
            "echo_std": float(vals.std()),
            "echo_entropy": float(measure.shannon_entropy(vals)),
            "echo_ratio_to_ring": float(vals.mean() / max(img[ring].mean(), 1e-6))}


def glcm_features(img, mask, levels=32):
    # skimage.feature.graycomatrix + graycoprops on the masked lesion; level 0 = outside mask
    q = (img.astype(float) / 256 * (levels - 1)).astype(np.uint8) + 1
    q[~mask] = 0
    m = feature.graycomatrix(q, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                             levels=levels + 1, symmetric=True, normed=False)
    m = m[1:, 1:].astype(float)
    m /= m.sum(axis=(0, 1), keepdims=True)
    return {f"glcm_{k}": float(feature.graycoprops(m, k).mean())
            for k in ("contrast", "homogeneity", "energy", "correlation")}


# --------------------------------------------------------------------------
# PyRadiomics
# --------------------------------------------------------------------------
def pyradiomics_2d(img, mask):
    from radiomics import featureextractor
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    ext = featureextractor.RadiomicsFeatureExtractor(force2D=True, force2DDimension=0,
                                                     label=1, binWidth=25)
    ext.disableAllFeatures()
    for cls in ("shape2D", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        ext.enableFeatureClassByName(cls)
    si = sitk.GetImageFromArray(img[None].astype(np.float32))
    sm = sitk.GetImageFromArray(mask[None].astype(np.uint8))
    t = time.perf_counter()
    out = ext.execute(si, sm)
    dt = time.perf_counter() - t
    feats = {k: float(v) for k, v in out.items() if not k.startswith("diagnostics")}
    return feats, dt


# --------------------------------------------------------------------------
# synthetic cases
# --------------------------------------------------------------------------
def synth_masks(n=256):
    circle = np.zeros((n, n), bool)
    rr, cc = draw.disk((100, 128), 50, shape=circle.shape)
    circle[rr, cc] = True
    star = np.zeros((n, n), bool)
    k = 12
    ang = np.linspace(0, 2 * np.pi, 2 * k, endpoint=False)
    rad = np.where(np.arange(2 * k) % 2 == 0, 70, 28)
    rr, cc = draw.polygon(100 + rad * np.sin(ang), 128 + rad * np.cos(ang), shape=star.shape)
    star[rr, cc] = True
    return circle, star


def synth_bmode(mask, posterior, rng, n=256):
    # speckle-like background ~120, hypoechoic lesion ~50, posterior region scaled
    img = rng.normal(120, 15, (n, n))
    img[mask] = rng.normal(50, 10, mask.sum())
    r0, c0, r1, c1 = _props(mask).bbox
    img[r1:r1 + (r1 - r0) // 2, c0:c1] *= posterior
    return np.clip(img, 0, 255).astype(np.uint8)


def all_features(img, mask):
    row = {
        "solidity": solidity(mask), "convexity": convexity(mask),
        "circularity": circularity(mask), "roundness": roundness(mask),
        "eccentricity": eccentricity(mask), "aspect_ratio": aspect_ratio(mask),
        "taller_than_wide": taller_than_wide(mask), "enc": enc(mask),
        "lobulations": lobulations(mask), "radial_roughness": radial_roughness(mask),
        "fractal_dim": fractal_dimension(mask),
        "margin_sharpness": margin_sharpness(img, mask), "edge_contrast": edge_contrast(img, mask),
        "posterior_ratio": posterior_acoustic_ratio(img, mask),
    }
    row.update(internal_echo(img, mask))
    row.update(glcm_features(img, mask))
    return row


def main():
    rng = np.random.default_rng(0)
    circle, star = synth_masks()
    img_c = synth_bmode(circle, 1.4, rng)   # posterior enhancement
    img_s = synth_bmode(star, 0.4, rng)     # posterior shadowing
    noise = rng.integers(0, 256, circle.shape).astype(np.uint8)
    smooth = np.tile(np.linspace(40, 200, 256), (256, 1)).astype(np.uint8)

    cases = {"circle(enh)": (img_c, circle), "star(shadow)": (img_s, star),
             "noise+circle": (noise, circle), "smooth+circle": (smooth, circle)}
    table = {}
    for name, (im, m) in cases.items():
        t = time.perf_counter()
        table[name] = all_features(im, m)
        table[name]["_ms"] = 1000 * (time.perf_counter() - t)

    keys = list(table["circle(enh)"].keys())
    print(f"{'feature':22s}" + "".join(f"{k:>15s}" for k in cases))
    for k in keys:
        print(f"{k:22s}" + "".join(f"{table[c][k]:15.3f}" for c in cases))

    c, s, nz, sm = (table[k] for k in cases)
    checks = {
        "circle circularity > 0.95": c["circularity"] > 0.95,
        "circle solidity > 0.98": c["solidity"] > 0.98,
        "star solidity < 0.8": s["solidity"] < 0.8,
        "circle convexity > 0.98 > star": c["convexity"] > 0.98 > s["convexity"],
        "circle eccentricity < 0.1": c["eccentricity"] < 0.1,
        "circle aspect/TTW ~ 1": abs(c["aspect_ratio"] - 1) < 0.05 and abs(c["taller_than_wide"] - 1) < 0.05,
        "circle ENC ~ 1 < star ENC": abs(c["enc"] - 1) < 0.05 and s["enc"] > 1.2,
        "circle lobulations 0, star 12": c["lobulations"] == 0 and s["lobulations"] == 12,
        "star roughness > circle": s["radial_roughness"] > 5 * c["radial_roughness"],
        "fractal dim circle ~1, star > circle": 0.9 < c["fractal_dim"] < 1.15 and s["fractal_dim"] >= c["fractal_dim"],
        "posterior ratio circle > 1 > star": c["posterior_ratio"] > 1.2 and s["posterior_ratio"] < 0.6,
        "lesion hypoechoic (ring ratio < 0.6)": c["echo_ratio_to_ring"] < 0.6,
        "GLCM contrast noise >> smooth": nz["glcm_contrast"] > 50 * sm["glcm_contrast"],
        "GLCM homogeneity noise < smooth": nz["glcm_homogeneity"] < sm["glcm_homogeneity"],
        "margin sharpness smooth < synthetic lesion": sm["margin_sharpness"] < c["margin_sharpness"],
    }
    print("\nsanity checks")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    # PyRadiomics on synthetic circle and star
    print("\nPyRadiomics shape2D on synthetic masks")
    for name, (im, m) in list(cases.items())[:2]:
        f, dt = pyradiomics_2d(im, m)
        print(f"  {name}: Sphericity={f['original_shape2D_Sphericity']:.3f} "
              f"Elongation={f['original_shape2D_Elongation']:.3f} "
              f"PerimeterSurfaceRatio={f['original_shape2D_PerimeterSurfaceRatio']:.3f} "
              f"glcm_Contrast={f['original_glcm_Contrast']:.2f} n={len(f)} t={dt:.2f}s")

    # real BUS-BRA case
    img = np.array(Image.open(REAL_IMAGE).convert("L"))
    with zipfile.ZipFile(REAL_ZIP) as z:
        mask = np.array(Image.open(io.BytesIO(z.read(REAL_MASK_MEMBER))).convert("L")) > 127
    print(f"\nreal case {REAL_IMAGE.name}: image {img.shape}, mask px {mask.sum()}")
    if mask.shape != img.shape:
        raise SystemExit(f"shape mismatch {mask.shape} vs {img.shape}")
    real = all_features(img, mask)
    for k, v in real.items():
        print(f"  {k:22s}{v:12.3f}")
    f, dt = pyradiomics_2d(img, mask)
    per_cls = {}
    for k in f:
        per_cls[k.split("_")[1]] = per_cls.get(k.split("_")[1], 0) + 1
    print(f"PyRadiomics real case: {len(f)} features in {dt:.2f}s; per class {per_cls}")
    # warm second call
    _, dt2 = pyradiomics_2d(img, mask)
    print(f"PyRadiomics real case second call: {dt2:.2f}s")


if __name__ == "__main__":
    main()
