"""Build data/knowledge/manifest.csv from raw/*/meta.json plus two non-PMC sources.
Usage: python3 -I build_manifest.py <knowledge_dir>"""
import csv, json, sys
from pathlib import Path

K = Path(sys.argv[1])
WHY = {
 "lex_yoon2016_pmc5040135": "Per-descriptor PPV table for every grayscale 5th-edition BI-RADS US descriptor (shape, orientation, margin, echo, posterior, calcification, associated), 4th vs 5th edition (n=80 masses).",
 "lex_badupeprah2024_pmc11070917": "Per-descriptor sensitivity/specificity/PPV/odds ratio for BI-RADS US descriptors and associated features including vascularity and skin changes (n=60, Ghana).",
 "lex_chang2026_pmc13266155": "Concise list of BI-RADS v2025 US lexicon changes: lobulated shape, mixed solid and cystic, posterior combined removed, non-mass lesions, micro/macrocalcifications, echogenic rind and pseudocapsule, lymph node criteria.",
 "lex_song2026_pmc13554624": "BI-RADS v2025 US key updates (glandular tissue component, non-mass lesion descriptors and malignancy rates, lymph node morphology) with Asian-practice perspective.",
 "lex_kaplan2026_pmc13361901": "Side-by-side table of 2013 5th edition vs v2025 descriptors for US (and mammography/MRI) with rationale and reported diagnostic performance, incl. elastography.",
 "lex_kim2015_pmc4644748": "Pictorial essay of benign lesions that look irregular and hypoechoic (fat necrosis, mastitis, scars, sclerosing adenosis, etc.): the false-positive side of suspicious descriptors.",
 "lex_uematsu2023_pmc10354162": "Definitions and categories of non-mass lesions on breast US (ductal/non-ductal hypoechoic areas, architectural distortion, multiple small cysts) and their malignancy relevance.",
 "lex_kim2025_pmc12683742": "Non-mass lesion descriptors (distribution, echo pattern, associated calcification, duct change, shadowing, vascularity) with literature PPV ranges and a diagnostic approach.",
 "lex_alshoabi2023_pmc10191155": "Odds ratios of BI-RADS-type US features (irregular shape, non-circumscribed margin, distortion, skin thickening, nipple retraction, node enlargement) for a correct malignant US call; cohort is all-malignant, so not a benign-vs-malignant PPV study.",
 "quant_wei2020_pmc7547332": "Defines compactness, elliptic compactness, radial distance spectrum and LBP/HOG/GLCM texture features for benign/malignant breast US classification.",
 "quant_sultan2015_pmc8298005": "Inter-observation variability of BI-RADS US features and its effect on a Bayesian CAD probability of malignancy: how reliable descriptor-level inputs are.",
 "quant_fleury2019_pmc6682836": "Computable BI-RADS-mapped features (ADEE, orientation, NumPeaks, AvgDistance, convex hull area difference, echogenicity, entropy, posterior shadow, size) with formulas and ML AUCs.",
 "quant_zhou2015_pmc4414937": "Contour features (tumor circularity, normalized radial length mean/SD, area ratio, roughness index, SDD) and how posterior acoustic shadowing biases them.",
 "quant_chen2026_pmc13005611": "Quantitative orientation angle of the mass long axis vs skin with a malignancy cut-off (>22.3 deg), compared with visual parallel/non-parallel assessment.",
 "quant_ye2026_pmc13002577": "Mask-derived circularity, solidity, eccentricity and extent with formulas and benign/malignant medians, aligned with BI-RADS shape/margin.",
 "quant_zhuang2021_pmc8154287": "Quantifies orientation, edge indistinctness, posterior shadowing (Hu moments) and shape complexity (fractal dimension) from segmented breast US lesions.",
 "quant_chen2015_pmc4355599": "Textural, morphological, ellipsoid-fitting and posterior acoustic feature definitions (3D breast US) related to tumor grade.",
 "quant_huang2018_pmc5857346": "Survey of US CAD features: GLCM texture, spiculation, depth-to-width ratio, elliptic-normalized circumference and skeleton, with definitions.",
 "quant_daoud2020_pmc7730057": "Lists 10+ handcrafted morphological features (form factor, roundness, aspect ratio, convexity, solidity, extent, undulation, compactness, NRL entropy/variance, ellipse-fit features) combined with deep features.",
 "rev_gu2022_pmc9428828": "Review of breast US radiomics workflow and feature families (shape, first/second/higher-order texture, peritumoral) for diagnosis, subtype, nodes, NAC response.",
 "rev_sun2025_pmc12832378": "Recent review of US radiomics for breast cancer (morphological, texture, first-order, wavelet, intra- and peritumoral features) with reported AUCs.",
 "rev_brunetti2022_pmc9818181": "Rapid review of AI in breast US from diagnosis to prognosis, covering CAD and deep learning feature use.",
}
LIC = {"cc by": "CC BY 4.0", "cc by-nc": "CC BY-NC 4.0", "cc by-nc-nd": "CC BY-NC-ND 4.0"}
rows = []
for mj in sorted(K.glob("raw/*/meta.json")):
    m = json.loads(mj.read_text())
    sid = m["source_id"]; pmc = m["pmcid"]
    xml = (mj.parent / f"{pmc}.xml").read_text(errors="ignore")
    lic = LIC.get(m.get("license"), m.get("license"))
    for v in ("3.0", "4.0"):
        pass
    import re
    mm = re.search(r"creativecommons\.org/licenses/([a-z-]+)/([0-9.]+)", xml)
    if mm:
        lic = "CC " + mm.group(1).upper() + " " + mm.group(2)
    elif "Creative Commons Attribution License" in xml:
        lic = "CC BY (version not stated)"
    au = [a.get("fullName", "") for a in m.get("authorList", {}).get("author", [])]
    authors = ", ".join(au[:3]) + (" et al." if len(au) > 3 else "")
    rows.append(dict(source_id=sid, title=m["title"].rstrip("."), authors=authors, year=m.get("pubYear", ""),
        venue=m.get("journalInfo", {}).get("journal", {}).get("title", ""), doi=m.get("doi", ""), pmcid=pmc,
        url=f"https://europepmc.org/article/PMC/{pmc}", license=lic, category=m["category"],
        why_relevant=WHY[sid], file_txt=f"raw/{sid}/{pmc}.txt"))
rows.append(dict(source_id="ibsi_zwanenburg2016_arxiv1612.07003", title="Image biomarker standardisation initiative (IBSI reference manual, v11)",
    authors="Zwanenburg A, Leger S, Vallieres M et al.", year="2016 (v11 2019)", venue="arXiv (preprint; companion paper Radiology 2020;295:328-338)",
    doi="10.48550/arXiv.1612.07003", pmcid="", url="https://arxiv.org/abs/1612.07003", license="CC BY 4.0", category="ibsi",
    why_relevant="Reference definitions and benchmark values for radiomics features (morphology, intensity, histogram, GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM) incl. 2D variants usable on a B-mode image + mask.",
    file_txt="raw/ibsi_zwanenburg2016_arxiv1612.07003/1612.07003v11.txt"))
rows.append(dict(source_id="docs_pyradiomics_features", title="PyRadiomics documentation: Radiomic Features",
    authors="van Griethuysen JJM, Fedorov A, Parmar C et al. (PyRadiomics community)", year="2026 (retrieved 2026-10-30, 'latest')", venue="pyradiomics.readthedocs.io",
    doi="", pmcid="", url="https://pyradiomics.readthedocs.io/en/latest/features.html", license="BSD 3-Clause (repository LICENSE.txt)", category="library_docs",
    why_relevant="Formulas and plain-language definitions of every PyRadiomics feature (first order, 2D/3D shape, GLCM, GLSZM, GLRLM, NGTDM, GLDM) the agent could request from a mask.",
    file_txt="raw/docs_pyradiomics_features/features.txt"))
order = {"lexicon": 0, "quantitative": 1, "ibsi": 2, "library_docs": 3, "radiomics_review": 4}
rows.sort(key=lambda r: (order[r["category"]], r["source_id"]))
with open(K / "manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
