"""Fetch Europe PMC metadata + fullTextXML for a list of 'category PMCID' lines.

Usage: python3 -I fetch_pmc.py <list.txt> <knowledge_dir>
Writes raw/<source_id>/<pmcid>.xml, <pmcid>.txt, meta.json. Downloaded files are only
parsed as data (xml.etree), never executed.
"""
import json, re, subprocess, sys, time, unicodedata, urllib.parse
from pathlib import Path
import xml.etree.ElementTree as ET

def curl(url):
    r = subprocess.run(["curl", "-sL", "--max-time", "120", "-A", "modality-bench-knowledge/1.0 (research)", url],
                       capture_output=True)
    return r.stdout

def meta(pmcid):
    q = urllib.parse.urlencode({"query": f"PMCID:{pmcid}", "format": "json", "resultType": "core"})
    d = json.loads(curl("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + q))
    return d["resultList"]["result"][0]

SKIP = {"ref-list", "table-wrap-foot", "fn-group", "ack", "xref"}
BLOCK = {"p", "title", "caption", "tr", "list-item", "sec", "abstract", "label", "table-wrap", "fig"}

def text_of(el, out, inrow=False):
    tag = el.tag.split("}")[-1]
    if tag in SKIP:
        if tag == "xref" and el.tail:
            out.append(el.tail.replace("\n", " ") if inrow else el.tail)
        return
    if tag == "title":
        out.append(" " if inrow else "\n\n## ")
    elif tag in BLOCK and not (inrow and tag == "p"):
        out.append("\n")
    if tag in ("td", "th"):
        out.append(" | ")
    if el.text:
        out.append(el.text.replace("\n", " ") if inrow else el.text)
    for c in el:
        text_of(c, out, inrow or tag in ("tr", "td", "th"))
    if tag in BLOCK and not (inrow and tag == "p"):
        out.append("\n")
    if el.tail:
        out.append(el.tail.replace("\n", " ") if inrow else el.tail)

def xml_to_txt(xml_bytes, m):
    root = ET.fromstring(xml_bytes)
    parts = []
    head = [f"TITLE: {m.get('title','')}", f"AUTHORS: {m.get('authorString','')}",
            f"JOURNAL: {m.get('journalInfo',{}).get('journal',{}).get('title','')} ({m.get('pubYear','')})",
            f"DOI: {m.get('doi','')}  PMCID: {m.get('pmcid','')}  LICENSE: {m.get('license','')}"]
    ab = root.find(".//abstract")
    body = root.find(".//body")
    if ab is not None:
        parts.append("\n\n# ABSTRACT\n"); text_of(ab, parts)
    if body is not None:
        parts.append("\n\n# BODY\n"); text_of(body, parts)
    for tw in root.findall(".//back//app") :
        text_of(tw, parts)
    t = "".join(parts)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n(?= \| )", "\n", t)
    t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
    t = re.sub(r"\[\s*[,–\-\s]*\]", "", t)
    return "\n".join(head) + "\n" + t.strip() + "\n", body is not None

def slug(m, cat):
    first = (m.get("authorList", {}).get("author", [{}])[0].get("lastName") or "anon")
    first = unicodedata.normalize("NFKD", first).encode("ascii", "ignore").decode().lower()
    first = re.sub(r"[^a-z]", "", first) or "anon"
    pre = {"lexicon": "lex", "quantitative": "quant", "radiomics_review": "rev", "ibsi": "ibsi"}[cat]
    return f"{pre}_{first}{m.get('pubYear','')}_{m['pmcid'].lower()}"

def main():
    lst, kdir = Path(sys.argv[1]), Path(sys.argv[2])
    for line in lst.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cat, pmcid = line.split()[:2]
        m = meta(pmcid); time.sleep(0.4)
        sid = slug(m, cat)
        d = kdir / "raw" / sid
        if (d / f"{pmcid}.txt").exists():
            print("skip", sid); continue
        d.mkdir(parents=True, exist_ok=True)
        x = curl(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"); time.sleep(0.4)
        if b"<article" not in x[:2000]:  # fallback: NCBI efetch (OA subset)
            x = curl(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid[3:]}&rettype=xml"); time.sleep(0.4)
            x = x.strip()
            if b"<pmc-articleset>" in x[:500]:
                x = x.split(b"<pmc-articleset>", 1)[1].rsplit(b"</pmc-articleset>", 1)[0].strip()
        (d / "meta.json").write_text(json.dumps({"category": cat, "source_id": sid, **m}, indent=1, ensure_ascii=False))
        ok = False
        if b"<article" in x[:2000] and b"</article>" in x[-200:]:
            (d / f"{pmcid}.xml").write_bytes(x)
            try:
                txt, ok = xml_to_txt(x, m)
                (d / f"{pmcid}.txt").write_text(txt)
            except ET.ParseError as e:
                print("parse error", pmcid, e)
        print(sid, m.get("license"), "fulltext" if ok else "NO-BODY", len(x))

if sys.argv[1] == "--reconvert":
    for x in Path(sys.argv[2]).glob("raw/*/PMC*.xml"):
        m = json.loads((x.parent / "meta.json").read_text())
        txt, ok = xml_to_txt(x.read_bytes(), m)
        x.with_suffix(".txt").write_text(txt); print(x.parent.name, ok, len(txt.split()))
else:
    main()
