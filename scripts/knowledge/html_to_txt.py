"""HTML -> plain text (data only, html.parser). Usage: python3 -I html_to_txt.py in.html out.txt [header]"""
import re, sys
from html.parser import HTMLParser
class P(HTMLParser):
    def __init__(s):
        super().__init__(); s.out = []; s.skip = 0; s.inmain = False
    def handle_starttag(s, t, a):
        a = dict(a)
        if t in ("script", "style", "nav", "footer"): s.skip += 1
        if t == "div" and a.get("itemprop") == "articleBody": s.inmain = True
        if t in ("p", "div", "li", "tr", "br", "dt", "dd", "pre"): s.out.append("\n")
        if t in ("h1", "h2", "h3", "h4"): s.out.append("\n\n## ")
        if t == "li": s.out.append("- ")
    def handle_endtag(s, t):
        if t in ("script", "style", "nav", "footer"): s.skip -= 1
        if t in ("p", "h1", "h2", "h3", "h4", "dd", "pre"): s.out.append("\n")
    def handle_data(s, d):
        if s.skip <= 0 and s.inmain: s.out.append(d)
p = P(); p.feed(open(sys.argv[1], encoding="utf-8").read())
t = "".join(p.out).replace("¶", "")
t = re.sub(r"[ \t]+", " ", t); t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
hdr = (sys.argv[3] + "\n") if len(sys.argv) > 3 else ""
open(sys.argv[2], "w").write(hdr + t.strip() + "\n"); print(len(t.split()), "words")
