"""Extract text from a PDF (data only) with pypdf. Usage: python3 -I pdf_to_txt.py in.pdf out.txt [header]"""
import re, sys
from pypdf import PdfReader
r = PdfReader(sys.argv[1])
pages = []
for i, p in enumerate(r.pages, 1):
    t = p.extract_text() or ""
    pages.append(f"\n\n[page {i}]\n" + t)
t = "".join(pages)
t = re.sub(r"[ \t]+", " ", t)
hdr = (sys.argv[3] + "\n") if len(sys.argv) > 3 else ""
open(sys.argv[2], "w").write(hdr + t.strip() + "\n")
print(len(r.pages), "pages", len(t.split()), "words")
