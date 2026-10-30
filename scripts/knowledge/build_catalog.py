"""Write data/knowledge/feature_catalog_draft.csv (hand-curated from the corpus, 2026-10-30).
Usage: python3 -I build_catalog.py <knowledge_dir>
Column in_our_data: 'yes' = computable or describable from one B-mode image + lesion mask;
'needs_pixel_spacing' = needs mm calibration we may not have; 'NO: ...' = needs data our datasets lack."""
import csv, sys
from pathlib import Path

Y = "lex_yoon2016_pmc5040135"; B = "lex_badupeprah2024_pmc11070917"; C = "lex_chang2026_pmc13266155"
S = "lex_song2026_pmc13554624"; KA = "lex_kaplan2026_pmc13361901"; K15 = "lex_kim2015_pmc4644748"
K25 = "lex_kim2025_pmc12683742"; U = "lex_uematsu2023_pmc10354162"; AL = "lex_alshoabi2023_pmc10191155"
WEI = "quant_wei2020_pmc7547332"; SUL = "quant_sultan2015_pmc8298005"; FL = "quant_fleury2019_pmc6682836"
ZH = "quant_zhou2015_pmc4414937"; CH26 = "quant_chen2026_pmc13005611"; YE = "quant_ye2026_pmc13002577"
ZU = "quant_zhuang2021_pmc8154287"; CH15 = "quant_chen2015_pmc4355599"; HU = "quant_huang2018_pmc5857346"
DA = "quant_daoud2020_pmc7730057"; GU = "rev_gu2022_pmc9428828"; SUN = "rev_sun2025_pmc12832378"
BR = "rev_brunetti2022_pmc9818181"; IB = "ibsi_zwanenburg2016_arxiv1612.07003"; PR = "docs_pyradiomics_features"
yo = "Yoon 2016 5th ed., n=80"; bp = "Badu-Peprah 2024, n=60"
BM, BI, DOP, EL, CL = "bmode_image+mask", "bmode_image", "doppler", "elastography", "clinical"
NO_D, NO_E, NO_C = "NO: needs Doppler", "NO: needs elastography", "NO: needs clinical/prior data"

R = []
def add(feature, cat, definition, meas, direction, effect, srcs, avail="yes"):
    R.append(dict(feature=feature, bi_rads_category=cat, definition=definition, measurable_from=meas,
                  suspicious_direction=direction, reported_ppv_or_effect=effect, source_ids=";".join(srcs),
                  in_our_data=avail))

# ---------------- BI-RADS mass descriptors ----------------
add("shape: oval", "shape", "Elliptical or egg-shaped mass, may include 2-3 gentle undulations.", BM,
    "oval -> less suspicious (benign-leaning)", f"PPV 10.8% (4/37) [{Y}, {yo}]", [Y, B, KA, C])
add("shape: round", "shape", "Spherical/circular mass whose anteroposterior and transverse diameters are about equal.", BM,
    "round -> intermediate; less common in both groups", f"PPV 12.5% (1/8) [{Y}, {yo}]", [Y, B, KA])
add("shape: lobulated (BI-RADS v2025)", "shape", "Oval contour with one or more indentations giving an undulating outline; reintroduced in v2025.", BM,
    "lobulated -> slightly more suspicious than oval", "", [C, KA, AL])
add("shape: irregular", "shape", "Mass that is neither round nor oval (nor lobulated).", BM,
    "irregular -> more suspicious; benign mimics exist (fat necrosis, mastitis, scar, sclerosing adenosis)",
    f"PPV 68.6% (24/35) [{Y}, {yo}]; PPV 73.8% [{B}, {bp}]", [Y, B, K15, AL, KA])
add("orientation: parallel", "orientation", "Long axis of the mass parallel to the skin line ('wider than tall').", BM,
    "parallel -> less suspicious", f"PPV 23.9% (11/46) [{Y}, {yo}]", [Y, B, KA])
add("orientation: non-parallel (taller-than-wide)", "orientation", "Long axis not parallel to the skin; anteroposterior dimension exceeds horizontal ('taller than wide').", BM,
    "non-parallel -> more suspicious", f"PPV 52.9% (18/34) [{Y}, {yo}]; PPV 85.7% [{B}, {bp}]", [Y, B, KA, CH26])
add("margin: circumscribed", "margin", "Well-defined, sharp margin with an abrupt transition between lesion and surrounding tissue over the whole contour.", BM,
    "circumscribed -> less suspicious; any non-circumscribed segment upgrades the margin", f"PPV 0.0% (0/23) [{Y}, {yo}]", [Y, B, AL, KA])
add("margin: indistinct", "margin", "No clear demarcation between all or part of the margin and surrounding tissue.", BM,
    "indistinct -> more suspicious than circumscribed", f"PPV 46.2% (6/13) [{Y}, {yo}]; PPV 28.6% [{B}, {bp}]", [Y, B, KA])
add("margin: angular", "margin", "Part or all of the margin forms sharp corners, often acute angles.", BM,
    "angular -> more suspicious", f"PPV 46.7% (7/15) [{Y}, {yo}]; PPV 70.0% [{B}, {bp}]", [Y, B, KA])
add("margin: microlobulated", "margin", "Margin with many short-cycle small undulations (scalloped).", BM,
    "microlobulated -> more suspicious", f"PPV 33.3% (4/12) [{Y}, {yo}]; PPV 50.0% [{B}, {bp}]", [Y, B, SUL, KA])
add("margin: spiculated", "margin", "Margin with sharp lines radiating from the mass.", BM,
    "spiculated -> most suspicious margin descriptor", f"PPV 70.6% (12/17) [{Y}, {yo}]; PPV 100% [{B}, {bp}]", [Y, B, KA])
add("lesion boundary: abrupt interface (4th ed., removed in 5th)", "margin", "Sharp demarcation between lesion and surrounding tissue without a transition zone.", BM,
    "abrupt -> less suspicious; descriptor dropped in 5th edition", f"PPV 30.9% (21/68) [{Y}, 4th ed., n=80]", [Y, B])
add("lesion boundary: echogenic halo (4th ed., removed in 5th)", "margin", "Echogenic transition zone bridging the lesion and surrounding tissue; absence has no diagnostic meaning.", BM,
    "echogenic halo -> more suspicious (also in abscess)", f"PPV 66.7% (8/12) [{Y}, 4th ed.]; PPV 61.5% [{B}, {bp}]", [Y, B])
add("echo pattern: anechoic", "echo", "No internal echoes (fluid, as in a simple cyst).", BM,
    "anechoic -> benign (cyst)", f"PPV 0.0% (0/2) [{Y}, {yo}]", [Y, B, KA])
add("echo pattern: hyperechoic", "echo", "Echogenicity greater than subcutaneous fat, equal to fibroglandular tissue.", BM,
    "hyperechoic -> usually benign (e.g. lipoma, fat necrosis)", f"PPV 0.0% (0/2) [{Y}, {yo}]", [Y, B, KA])
add("echo pattern: complex cystic and solid", "echo", "Mass with both anechoic (cystic) and echogenic (solid) components; v2025 keeps it for thick-walled/septated cysts or cysts with solid nodules.", BM,
    "solid component -> more suspicious than pure cyst", f"PPV 0.0% (0/8) 5th ed.; 38.5% reported for 4th-ed. 'complex' [{Y}, n=80]", [Y, KA, C])
add("echo pattern: mixed solid and cystic (BI-RADS v2025)", "echo", "Mass clearly containing both solid and cystic components; emphasises the solid part.", BM,
    "larger/irregular solid component -> more suspicious", "", [C, KA])
add("echo pattern: hypoechoic", "echo", "Echogenicity lower than subcutaneous fat.", BM,
    "hypoechoic -> common in both benign and malignant; weak discriminator alone",
    f"PPV 42.5% (17/40) [{Y}, {yo}]; PPV 57.8% [{B}, {bp}]", [Y, B, AL])
add("echo pattern: markedly hypoechoic", "echo", "Very low echogenicity, darker than typical hypoechoic solid tissue.", BM,
    "markedly hypoechoic -> more suspicious", f"PPV 80.0% [{B}, {bp}]", [B])
add("echo pattern: isoechoic", "echo", "Echogenicity equal to subcutaneous fat.", BM,
    "isoechoic -> usually benign-leaning", f"PPV 0.0% (0/3) [{Y}, {yo}]", [Y, B])
add("echo pattern: heterogeneous", "echo", "Mixture of echogenic patterns within a solid mass.", BM,
    "heterogeneous -> raises suspicion mainly with irregular shape / non-circumscribed margin", f"PPV 48.0% (12/25) [{Y}, {yo}]", [Y, KA, AL])
add("posterior features: none", "posterior", "No shadowing or enhancement deep to the mass.", BM,
    "neutral", f"PPV 33.3% (11/33) [{Y}, {yo}]", [Y, B])
add("posterior features: enhancement", "posterior", "Increased echoes deep to the mass (good sound transmission, e.g. cysts, many fibroadenomas, some high-grade cancers).", BM,
    "enhancement -> benign-leaning but seen in high-grade tumours", f"PPV 6.7% (1/15) [{Y}, {yo}]", [Y, B, CH15])
add("posterior features: shadowing", "posterior", "Decreased echoes deep to the mass (attenuation), excluding edge refraction shadows.", BM,
    "shadowing -> more suspicious (also scars, macrocalcification); v2025: any shadowing is significant",
    f"PPV 26.7% (4/15) [{Y}, {yo}]; PPV 50.0% [{B}, {bp}]", [Y, B, C, ZH, FL])
add("posterior features: combined pattern", "posterior", "Both shadowing and enhancement deep to the mass; Chang 2026 reports it removed in v2025, Kaplan 2026 table lists it preserved.", BM,
    "combined -> suspicious", f"PPV 76.5% (13/17) [{Y}, {yo}]", [Y, C, KA])
# ---------------- calcifications ----------------
add("calcifications in a mass", "calcification", "Echogenic foci inside a mass (microcalcifications most visible inside hypoechoic masses).", BI,
    "present -> more suspicious", f"PPV 81.3% (13/16) [{Y}, {yo}]; PPV 66.7% [{B}, {bp}]", [Y, B, C])
add("calcifications outside a mass", "calcification", "Echogenic foci in fat or fibroglandular tissue outside a mass (may lie in a non-mass lesion).", BI,
    "present -> moderately suspicious; less conspicuous than on mammography", f"PPV 100% (1/1) [{Y}, {yo}]", [Y, C])
add("intraductal calcifications", "calcification", "Echogenic foci located within a duct (added in 5th edition).", BI,
    "present -> suspicious (DCIS)", f"PPV 100% (3/3) [{Y}, {yo}]", [Y, C])
add("macrocalcification vs microcalcification (v2025)", "calcification", "Macro: large enough to attenuate the beam, usually shadowing (degenerating fibroadenoma, oil cyst); micro: tiny echogenic foci without shadow.", BI,
    "microcalcification in hypoechoic mass -> suspicious; isolated macrocalcification -> usually benign", "", [C, KA])
# ---------------- associated features ----------------
add("architectural distortion", "associated", "Disruption of surrounding tissue planes by the lesion (compression, obliteration of planes, thickened/straightened Cooper ligaments).", BI,
    "present -> more suspicious", f"PPV 68.8% (11/16) [{Y}, {yo}]; PPV 71.4% [{B}, {bp}]", [Y, B, U, AL])
add("duct changes", "associated", "Abnormal calibre and/or arborisation of a duct, extending to or from the mass, or intraductal mass/thrombus.", BI,
    "abnormal duct change -> more suspicious; solitary dilated duct malignancy up to 9.5%", f"PPV 54.5% (6/11) [{Y}, {yo}]; solitary dilated duct malignancy rate 9.5% [{KA}]", [Y, KA, K25])
add("skin thickening", "associated", "Focal or diffuse skin thickening >2 mm (except periareolar/inframammary).", BI,
    "present -> more suspicious (inflammatory carcinoma, but also mastitis, post-radiation)", f"PPV 66.7% [{B}, {bp}]; 0/0 in [{Y}]", [Y, B, AL], "yes (only if skin is in the image)")
add("skin retraction / nipple retraction", "associated", "Skin surface concave or ill-defined and pulled in.", BI,
    "present -> more suspicious", f"Adjacent tissue retraction PPV 50.0% [{B}]; nipple retraction OR 7.5 for malignant US call [{AL}]", [Y, B, AL], "yes (only if skin is in the image)")
add("edema", "associated", "Increased echogenicity of surrounding tissue with reticular network of hypoechoic lines (lymphatics/fluid).", BI,
    "present -> more suspicious (also infection, radiation)", f"PPV 40.0% (2/5) [{Y}, {yo}]; skin edema PPV 76.5% [{B}]", [Y, B])
add("adjacent tissue infiltration", "associated", "Lesion tissue extending into surrounding fat or parenchyma (study-specific term).", BI,
    "present -> more suspicious", f"PPV 80.0% [{B}, {bp}]", [B])
add("echogenic rind (BI-RADS v2025)", "associated", "Thick, non-uniform echogenic band surrounding a lesion of any shape (peritumoral tissue reaction).", BM,
    "present -> more suspicious, more specific than sensitive; also fat necrosis",
    f"hyperechoic zone sens 87%, spec 81% (137 lesions, cited in [{C}])", [C, KA])
add("echogenic pseudocapsule (BI-RADS v2025)", "associated", "Thin, uniform echogenic line surrounding (usually oval) mass.", BM,
    "present around oval/lobulated mass without other suspicious features -> likely benign", "", [C])
add("vascularity: absent / internal / peripheral hypervascularity (rim)", "associated", "Colour/power Doppler flow pattern; v2025 renames 'vessels in rim' to 'peripheral hypervascularity'.", DOP,
    "internal or peripheral hypervascularity -> more suspicious; absent -> not reassuring by itself",
    f"vascularity PPV 67.7% [{B}, {bp}]; NML hypervascularity PPV 17.0-81.7% [{K25}]", [B, C, KA, K25], NO_D)
add("elasticity assessment: soft / intermediate / hard", "associated", "Tissue stiffness from strain or shear-wave elastography.", EL,
    "hard -> more suspicious; must not overrule grayscale features",
    f"strain elastography + B-mode sens 99.0%, spec 91.5%; SWE AUC 0.928 vs 0.851 B-mode alone [{KA}]", [KA, Y], NO_E)
# ---------------- non-mass lesions (v2025) ----------------
add("non-mass lesion (finding type, v2025)", "other", "Focal area distinguishable from surrounding tissue that does not meet mass criteria (not seen as a 3D space-occupying lesion in two planes); shape/margin do not apply.", BM,
    "NML with linear/segmental distribution, duct change, calcification or shadowing -> suspicious",
    f"malignancy 10.4% screening, 43.4% diagnostic, 40.1% current cancer [{S}]; 2.2-8.0% screening vs 16.1-53.8% diagnostic [{K25}]", [C, S, KA, K25, U])
add("NML distribution: focal / linear / segmental / regional / diffuse", "other", "Spatial extent of a non-mass lesion; linear = along a duct, segmental = triangular towards nipple (one duct system).", BM,
    "linear or segmental -> more suspicious", f"segmental PPV 22.9% [{C}]; segmental/linear PPV 22.9-76.6% [{K25}]", [C, KA, K25])
add("NML echo pattern: anechoic / hypoechoic / heterogeneous (mixed) / hyperechoic", "echo", "Internal echogenicity of a non-mass lesion.", BM,
    "mixed/heterogeneous -> somewhat more suspicious (inconsistent across studies)", f"mixed echogenicity PPV 16.6% [{C}; {K25}]", [C, KA, K25])
add("NML associated: calcifications", "calcification", "Echogenic foci within a non-mass lesion.", BI,
    "present -> more suspicious", f"PPV 44% [{C}]; 44.3-100% [{K25}]", [C, K25])
add("NML associated: posterior shadowing", "posterior", "Shadowing deep to a non-mass area.", BI,
    "present -> more suspicious", f"PPV 22% [{C}]; 21.8-95.0% [{K25}]", [C, K25])
add("NML associated: small interspersed cysts", "other", "Non-mass area with multiple small cysts and no suspicious features (fibrocystic change).", BI,
    "small cysts without other features -> benign-leaning", f"PPV 4.1-36.8% [{K25}]", [K25, C, U])
add("NML associated: architectural distortion", "associated", "Distortion of fibroglandular tissue by a non-mass lesion.", BI,
    "present -> suspicious but inconsistent association", f"PPV 10.0-100%, often not significant [{K25}]", [U, K25])
# ---------------- special cases / lymph nodes / tissue ----------------
add("special case: simple cyst", "other", "Circumscribed, anechoic, oval/round mass with posterior enhancement and thin wall.", BM, "typically benign (BI-RADS 2)", "", [C, KA])
add("special case: clustered microcysts / complicated cyst", "other", "Cluster of tiny anechoic foci with thin septa / cyst with homogeneous low-level internal echoes.", BM, "usually benign or probably benign", "", [C, KA])
add("special case: mass in or on skin; foreign body incl. implants; post-surgical fluid collection; fat necrosis", "other", "Lexicon special cases with characteristic appearance (sebaceous cyst, implant, seroma, fat necrosis).", BI,
    "usually benign but need clinical history", "", [C, KA, K15], "partial (history needed)")
add("special case: post-traumatic changes and abscess (v2025)", "other", "New v2025 special cases; abscess shows thick echogenic rim, internal debris and clinical signs of infection.", BI,
    "benign but irregular and can mimic cancer; needs clinical context", "", [C, K15], "partial (history needed)")
add("intramammary / axillary lymph node morphology", "other", "Node size, shape, cortical thickness, margin, fatty hilum, echogenic foci, symmetry, interval change (v2025 lymph-node category).", BI,
    "cortex >3 mm, eccentric cortical bulge, effaced/absent fatty hilum, round shape, non-circumscribed margin -> suspicious for metastasis",
    f"normal cortex <=3 mm [{C}; {KA}]; node enlargement OR 5.7 for malignant US call [{AL}]", [C, S, KA, AL], "partial (only if node is imaged)")
add("tissue composition: background echotexture / glandular tissue component (GTC, v2025)", "other", "Whole-breast background (homogeneous fat, homogeneous fibroglandular, heterogeneous) and share of glandular tissue (minimal/mild/moderate/marked).", BI,
    "heterogeneous background lowers lesion conspicuity; high GTC -> higher cancer risk (risk factor, not lesion feature)",
    f"high vs low GTC HR 1.5 (95% CI 1.05-2.1) [{C}]", [C, S], "partial (needs whole-breast view, not a lesion crop)")
add("lesion size (maximum diameter)", "other", "Largest diameter of the lesion in mm.", BM, "larger -> more often malignant in some studies; mainly for staging and follow-up",
    "", [FL, CH26, PR], "needs_pixel_spacing")
add("clinical context: patient age, symptoms/palpability, mammographic correlate, interval growth", "other", "Non-image information radiologists use with the lexicon (screening vs diagnostic, prior exams).", CL,
    "older age, palpable/symptomatic, suspicious mammographic correlate, new or growing -> more suspicious",
    f"NML malignancy 2.8% with negative vs 28.8% with positive mammogram [{C}]; screening 10.4% vs diagnostic 43.4% [{S}]", [C, S, KA], NO_C)
# ---------------- quantitative morphology (mask) ----------------
add("circularity (2D)", "shape", "4*pi*Area/Perimeter^2; 1 for a perfect circle.", BM, "lower -> more irregular -> more suspicious",
    f"median benign 0.85 vs malignant 0.50 (BUS dataset) [{YE}]; half-contour TC AUC 0.78 in shadowed tumours [{ZH}]", [YE, ZH, DA])
add("solidity", "shape", "Area / convex hull area.", BM, "lower -> concavities (lobulation/spiculation) -> more suspicious",
    f"median benign 0.94 vs malignant 0.80 [{YE}]", [YE, DA])
add("convexity", "shape", "Convex hull perimeter / lesion perimeter.", BM, "lower -> more irregular margin -> more suspicious", "", [DA])
add("extent (rectangularity)", "shape", "Area / bounding-box area.", BM, "lower -> more suspicious", f"median benign 0.75 vs malignant 0.55 [{YE}]", [YE, DA])
add("eccentricity", "shape", "Eccentricity of the ellipse with the same second moments (0 circle, ->1 elongated).", BM,
    "higher -> more suspicious in Ye 2026 (direction depends on orientation; benign fibroadenomas are also elongated but parallel)",
    f"median benign 0.45 vs malignant 0.70 [{YE}]", [YE])
add("depth-to-width ratio / aspect ratio", "orientation", "Max vertical extent / max horizontal extent of the lesion (image axes).", BM,
    ">1 (taller than wide) -> more suspicious", "malignant usually >1, benign <1 [" + HU + "]", [HU, DA, FL])
add("long axis-to-short axis ratio (L:S)", "shape", "Major / minor axis of the equivalent ellipse.", BM, "higher -> more elongated; read together with orientation (parallel elongated masses are often fibroadenomas)", "", [HU, DA])
add("orientation angle", "orientation", "Angle between the lesion's long axis and the skin line.", BM,
    "larger angle -> more suspicious", f"cut-off >22.3 deg: sens 84.9%, spec 82.8%; AUC 0.838 vs 0.651 visual [{CH26}]", [CH26, FL, ZU])
add("compactness", "shape", "Perimeter^2/Area-type measure of similarity to a circle (several definitions; inverse of circularity).", BM,
    "further from circle -> more suspicious", "", [WEI, DA, CH15, IB])
add("elliptic-normalized circumference (ENC) / elliptic compactness", "shape", "Lesion perimeter relative to the perimeter of its fitted (equivalent) ellipse.", BM,
    "perimeter much larger than ellipse perimeter -> lobulated/spiculated -> more suspicious", "", [HU, WEI, DA])
add("elliptic-normalized skeleton (ENS)", "shape", "Number of skeleton points of the lesion normalised by the perimeter of the equivalent ellipse.", BM,
    "more skeleton branches -> more irregular -> more suspicious", "", [HU])
add("area difference with equivalent ellipse (ADEE)", "shape", "(A_ellipse + A_tumour - A_intersection) / A_tumour.", BM,
    "higher -> shape deviates from ellipse -> more suspicious", f"part of 10-feature set, SVM AUC 0.840 [{FL}]", [FL])
add("area difference with convex hull (ADCH)", "margin", "(A_convexhull - A_tumour) / A_tumour.", BM,
    "higher -> irregular/angular contour -> more suspicious", f"part of 10-feature set, SVM AUC 0.840 [{FL}]", [FL])
add("number of contour peaks (NumPeaks) / lobulation count", "margin", "Local maxima of the distance between contour and convex hull (counts lobulations).", BM,
    "more peaks -> bumpier (microlobulated) margin -> more suspicious", "", [FL, DA])
add("average distance to convex hull (AvgDistance)", "margin", "Mean distance between contour and convex hull.", BM, "higher -> spiculated contour -> more suspicious", "", [FL])
add("normalized radial length (NRL) mean / SD / entropy", "margin", "Distances from centroid to contour points normalised by the maximum; statistics describe boundary roughness.", BM,
    "higher SD/entropy -> irregular margin -> more suspicious", "", [ZH, DA])
add("roughness index", "margin", "Average variation between neighbouring normalized radial lengths, a measure of margin spiculation (Zhou 2015).", BM, "higher -> more suspicious", "", [ZH])
add("standard deviation of degree (SDD) of contour", "margin", "Variability of local contour direction along the (half) contour.", BM,
    "higher -> irregular -> more suspicious; robust to posterior shadowing when computed on the upper half contour",
    f"half-contour SDD AUC 0.81 vs full-contour 0.61 in shadowed tumours [{ZH}]", [ZH])
add("area ratio (contour)", "shape", "Percentage of tumour area lying outside the circle defined by the mean radial length (Zhou 2015).", BM, "higher -> more irregular -> more suspicious", "", [ZH])
add("spiculation (frequency-domain measure)", "margin", "Ratio of low- to high-frequency area of the contour (radial) signal spectrum.", BM, "Huang 2018: value reported as proportional to malignancy likelihood", "", [HU])
add("undulation characteristics", "margin", "Number/size of contour undulations (classic CAD definition relative to the maximum inscribed circle; Daoud 2020 lists it without formula).", BM, "more undulations -> more suspicious", "", [DA])
add("fractal dimension of contour (shape complexity)", "margin", "Divider/box-counting fractal dimension of the lesion boundary.", BM, "higher -> more complex margin -> more suspicious", "", [ZU])
add("form factor / roundness", "shape", "Form factor 4*pi*A/P^2; roundness 4*A/(pi*major_axis^2) (common CAD definitions; Daoud 2020 lists names only).", BM, "lower -> more suspicious", "", [DA])
add("ellipse-fit features (axis lengths, axis ratio, major-axis angle, ellipse-tumour overlap)", "shape", "Parameters of the best-fit ellipse and its agreement with the lesion.", BM,
    "low overlap, long axis non-parallel to skin -> more suspicious", "", [DA, CH15, WEI])
add("2D shape set (PyRadiomics/IBSI): mesh area, perimeter, perimeter-to-surface ratio, sphericity, max 2D diameter, major/minor axis length, elongation", "shape",
    "Standardised 2D shape descriptors from the mask.", BM, "less round/compact -> more suspicious (no fixed thresholds)", "", [PR, IB], "yes (mm units need pixel spacing)")
add("edge indistinctness / margin sharpness (gradient across boundary)", "margin", "Intensity gradient or contrast across the lesion boundary band.", BM,
    "blurred (low gradient) margin -> indistinct -> more suspicious", "", [ZU, FL, HU])
# ---------------- quantitative intensity / posterior ----------------
add("echogenicity (lesion vs surrounding mean intensity)", "echo", "Mean grey level inside the mask relative to surrounding tissue or fat.", BM,
    "darker than surrounding tissue (markedly hypoechoic) -> more suspicious", "", [FL, CH15], "yes (gain/TGC dependent)")
add("posterior acoustic difference (Shadow = mean I_posterior - mean I_tumour)", "posterior", "Mean intensity of a same-size region beneath the lesion minus mean lesion intensity; negative = shadow.", BM,
    "negative (shadowing) -> more suspicious; positive (enhancement) -> benign-leaning", "", [FL, CH15, ZU, ZH])
add("posterior region statistics / Hu moments of posterior region", "posterior", "Std, mean, or shape moments of the region behind the lesion.", BM, "darker/heterogeneous posterior region -> shadowing", "", [CH15, ZU])
add("internal heterogeneity entropy (neighbour-difference entropy)", "texture", "Entropy of the distribution of intensity differences between adjacent pixels inside the lesion.", BM,
    "weak alone (RF AUC 0.481 [" + FL + "]); higher -> more heterogeneous", "", [FL])
add("first-order intensity features (mean, variance, skewness, kurtosis, entropy, energy, percentiles)", "texture", "Statistics of the grey-level histogram inside the mask.", BM,
    "study-specific; no fixed direction", "", [PR, IB, GU, SUN], "yes (gain/scanner dependent)")
add("GLCM features (contrast, correlation, energy, homogeneity, entropy, cluster shade/prominence, IMC1/2, IDN, IDMN)", "texture", "Second-order statistics of how often grey-level pairs co-occur at given offsets/angles.", BM,
    "study-specific; malignant lesions often more heterogeneous", "", [PR, IB, DA, WEI, HU, GU])
add("GLRLM / GLSZM / GLDM / NGTDM features", "texture", "Higher-order texture: run lengths, zone sizes, dependence and neighbourhood tone differences of grey levels.", BM,
    "study-specific", "", [PR, IB, GU, SUN])
add("local binary patterns (LBP)", "texture", "Histogram of binary codes comparing each pixel with its neighbours (micro-texture).", BM, "study-specific", "", [WEI])
add("histogram of oriented gradients (HOG)", "texture", "Histogram of local gradient directions in image cells (edge/structure layout).", BM, "study-specific", "", [WEI])
add("wavelet / LoG-filtered radiomic features", "texture", "First-order and texture features computed on wavelet or Laplacian-of-Gaussian filtered images.", BM, "study-specific", "", [SUN, PR, IB])
add("peritumoral radiomic features", "texture", "Radiomic features from a ring (e.g. 3 mm) around the mask.", BM,
    "adds to intratumoral features", f"intratumoral AUC 0.780 -> 0.840 with peritumoral [{SUN}]", [SUN, GU], "needs_pixel_spacing (ring width in mm)")
add("quantitative ultrasound (QUS) spectral / backscatter parameters", "texture", "Parameters from raw RF data (spectral slope, intercept, Nakagami etc.).", "rf_data",
    "study-specific", "", [CH15, HU], "NO: needs raw RF data")
add("deep learning features (CNN/ViT embeddings)", "texture", "Learned image representations, optionally combined with handcrafted features.", BI,
    "not interpretable per se", f"VGG19 conv features + morphology accuracy 96.1% vs 94.2% alone [{DA}]", [DA, BR, GU, SUN])

K = Path(sys.argv[1])
with open(K / "feature_catalog_draft.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(R[0].keys())); w.writeheader(); w.writerows(R)
print(len(R), "rows")
