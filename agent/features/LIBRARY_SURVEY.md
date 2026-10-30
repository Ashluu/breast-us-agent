# Feature Library Survey (B-mode image + binary lesion mask)

Date: 2026-10-30. Code: `scripts/feature_library_survey.py`. Env: conda `breast-agent`
(Python 3.11.18, macOS arm64). Pinned versions: `agent/requirements-rag.txt`.

## Environment commands (as run)

```bash
# 1. base env, scientific stack from conda-forge
~/miniforge3/bin/conda create -y -n breast-agent -c conda-forge python=3.11 \
    numpy scipy pandas scikit-learn scikit-image simpleitk matplotlib pillow lxml requests pyyaml pip
P=~/miniforge3/envs/breast-agent/bin/python

# 2. pyradiomics: no osx-arm64 wheel on PyPI, no conda-forge package, no Xcode CLT here.
#    `pip install pyradiomics`                       -> FAILS (setup.py imports numpy in isolated build)
#    `pip install --no-build-isolation pyradiomics`  -> FAILS (needs versioneer, then no C compiler)
#    What worked: zig as the C compiler (pip package `ziglang`, wrapper script below)
$P -m pip install setuptools wheel versioneer ziglang
cat > zcc <<'EOF'
#!/bin/bash
args=(); skip=0
for a in "$@"; do
  if [ $skip = 1 ]; then skip=0; continue; fi
  case "$a" in -arch) skip=1;; -fdebug-prefix-map=*) ;; *) args+=("$a");; esac
done
exec ~/miniforge3/envs/breast-agent/bin/python -m ziglang cc -target aarch64-macos "${args[@]}"
EOF
chmod +x zcc
CC=$PWD/zcc LDSHARED="$PWD/zcc -shared -Wl,-undefined,dynamic_lookup" \
    $P -m pip install --no-build-isolation --no-cache-dir pyradiomics
#    (with "-bundle -undefined dynamic_lookup" the zig linker segfaults; "-shared" works)
#    The installed package reports 3.0.1, built from pyradiomics-3.1.0.tar.gz (version bug in the sdist).

# 3. torch from conda-forge, NOT pip: pip torch bundles its own libomp and the process
#    aborts with "OMP: Error #15" next to conda-forge numpy/scikit-learn.
~/miniforge3/bin/conda install -y -n breast-agent -c conda-forge pytorch torchvision

# 4. everything else from pip
$P -m pip install timm transformers langgraph langchain-core langchain-ollama rank-bm25 \
    pypdf opencv-python-headless pyarrow pydicom gradio sentence-transformers openpyxl

# 5. retriever models
ollama pull nomic-embed-text
# MedCPT Query/Article/Cross encoders are downloaded on first from_pretrained() into ~/.cache/huggingface
```

## Survey Table

Library column: the function that does the work. "Composite" means no single library call
exists; it is built from 3-10 lines of library calls (in the survey script).
Synthetic tests: 256x256, circle r=50 with posterior enhancement (x1.4), 12-point star
(radii 70/28) with posterior shadowing (x0.4), hypoechoic lesion (mean 50) on background
(mean 120, sd 15). Texture tests also on uniform noise and a smooth ramp.

| Feature | Library function(s) | Direct? | Circle | Star | Real 0605-r | Check |
|---|---|---|---|---|---|---|
| Solidity | `skimage.measure.regionprops(...).solidity` | Yes | 0.989 | 0.425 | 0.890 | PASS |
| Convexity (hull perim / perim) | `cv2.convexHull` + `cv2.arcLength`, `regionprops.perimeter_crofton` | Composite | 1.002 | 0.417 | 0.870 | PASS |
| Circularity 4πA/P² | `regionprops.area`, `regionprops.perimeter_crofton` | Composite | 1.009 | 0.073 | 0.596 | PASS |
| Roundness 4A/(π·major²) | `regionprops.axis_major_length` | Composite | 1.000 | 0.765 | 0.689 | PASS |
| Eccentricity | `regionprops.eccentricity` | Yes | 0.000 | 0.000 | 0.703 | PASS (star is rotationally symmetric, so 0 is correct) |
| Aspect ratio (major/minor) | `regionprops.axis_major_length / axis_minor_length` (alt. `cv2.minAreaRect`) | Composite | 1.000 | 1.000 | 1.407 | PASS |
| Taller-than-wide (h/w) | `regionprops.bbox` | Composite | 1.000 | 1.000 | 0.645 | PASS |
| ENC (elliptic-normalized circumference) | `cv2.fitEllipse` + Ramanujan perimeter, `perimeter_crofton` | Composite | 1.005 | 2.975 | 1.233 | PASS |
| Lobulations | `cv2.convexityDefects` (depth > 5% of equivalent diameter) | Yes (+threshold) | 0 | 12 | 6 | PASS |
| Boundary roughness (radial sd/mean) | `cv2.findContours`, `regionprops.centroid` | Composite | 0.006 | 0.245 | 0.157 | PASS |
| Fractal dimension (box counting) | none in skimage/cv2; numpy `add.reduceat` | Composite | 0.969 | 1.221 | 1.030 | PASS |
| Margin sharpness (gradient on ±3 px ring) | `skimage.filters.sobel`, `scipy.ndimage.distance_transform_edt` | Composite | 0.098 | 0.094 | 0.077 | PASS (smooth ramp 0.003) |
| Edge contrast (outside − inside ring) | `scipy.ndimage.distance_transform_edt` | Composite | 75.5 | 69.4 | 26.5 | PASS |
| Posterior acoustic ratio | none; bbox slicing below lesion vs lateral strips | Composite | 1.399 | 0.398 | 1.145 | PASS (enhancement > 1 > shadowing) |
| Internal echo mean / sd / entropy | `regionprops(intensity_image).intensity_mean`, `skimage.measure.shannon_entropy` | Yes | 49.3 / 10.1 / 5.37 | 49.5 / 9.9 / 5.34 | 121.4 / 41.7 / 7.34 | PASS |
| Echogenicity vs surrounding ring | `scipy.ndimage.binary_dilation` | Composite | 0.390 | 0.419 | 0.754 | PASS (hypoechoic < 1) |
| GLCM contrast / homogeneity / energy / correlation | `skimage.feature.graycomatrix`, `graycoprops` (masked: level 0 = outside, dropped) | Yes | 3.20 / 0.47 / 0.22 / 0.00 | 3.12 / 0.48 / 0.22 / 0.00 | 4.68 / 0.47 / 0.10 / 0.91 | PASS (noise 159.7 vs ramp 0.07) |
| PyRadiomics shape2D (9) | `radiomics.featureextractor.RadiomicsFeatureExtractor(force2D=True)` | Yes | Sphericity 0.952 | Sphericity 0.256 | 9 features | PASS |
| PyRadiomics firstorder / glcm / glrlm / glszm / gldm / ngtdm | same extractor, `enableFeatureClassByName` | Yes | | | 18 / 24 / 16 / 16 / 14 / 5 | runs |

All 15 sanity checks pass.

## PyRadiomics On A Real Case

- Image `data/bus_bra/images/bus_0605-r.png` (433x322), mask `BUSBRA/Masks/mask_0605-r.png`
  read straight from `data/bus_bra/BUSBRA.zip` (24,473 lesion pixels).
- Settings: original image only, `force2D=True`, `force2DDimension=0`, `binWidth=25`, label 1,
  classes shape2D, firstorder, glcm, glrlm, glszm, gldm, ngtdm.
- Result: **102 features in 0.03-0.04 s** per case (CPU).
- The C extension was compiled with zig. Its outputs look right: circle Sphericity 0.952
  vs star 0.256, and the extraction completes on the real case.

## Retriever Check

| Model | Device | Dim | Time (warm, batch of 2) |
|---|---|---|---|
| ncbi/MedCPT-Query-Encoder | MPS / CPU | 768 | 8.6 / 13.9 ms per text |
| ncbi/MedCPT-Article-Encoder | MPS / CPU | 768 | 8.3 / 18.5 ms per text |
| ncbi/MedCPT-Cross-Encoder | CPU | 1 logit | 29.3 ms per pair |
| nomic-embed-text (Ollama `/api/embed`) | Ollama | 768 | 18.6 ms for 2 texts (warm; 6.2 s cold load) |

Sanity: in the query x article dot-product matrix each query scores highest on its matching
article ([[64.1, 58.4], [60.0, 63.0]]). The cross-encoder ranks the spiculated-margin article
above the fibroadenoma article for the shadowing query (-2.57 vs -6.22). Each MedCPT
safetensors file is 418 MB. huggingface_hub 2.4 stores the weights in a shared `~/.cache/huggingface/hub/blobs/`.

## Notes

- `cv2.arcLength` on a raw pixel contour (`CHAIN_APPROX_NONE`) overcounts the staircase:
  a circle gives convexity 0.958 and ENC 1.050. Using `regionprops.perimeter_crofton` as the
  lesion perimeter fixes both (1.002 and 1.005). Use the Crofton perimeter in any shape ratio.
- OpenCV 5 returns `convexityDefects` as (N, 4), while OpenCV 4 returns (N, 1, 4). Use `reshape(-1, 4)`.
- PyRadiomics texture values depend on `binWidth`. On the synthetic lesion (sd 10) a bin
  width of 25 leaves only 2-3 grey levels, so circle and star give the same glcm Contrast
  (0.55). On real B-mode images (sd ~40) this is less of an issue, but fix `binWidth`
  (or `binCount`) once and keep it the same everywhere.
- Mostly composites: of the BI-RADS-style descriptors, only solidity, eccentricity,
  intensity stats, GLCM and the PyRadiomics classes are one-line library calls. Posterior
  acoustic ratio, ENC, fractal dimension and margin sharpness have no library function.
  Their definitions (band widths, region sizes, thresholds) are our choices and should be
  stated in the thesis.
- `skimage.morphology.binary_dilation` is deprecated in scikit-image 0.26. Use `scipy.ndimage.binary_dilation`.
