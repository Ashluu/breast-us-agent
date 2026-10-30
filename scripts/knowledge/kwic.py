"""Print sentences matching a regex. Usage: python3 -I kwic.py REGEX file... [--max N]"""
import re, sys
args = sys.argv[1:]; mx = 15
if "--max" in args:
    i = args.index("--max"); mx = int(args[i+1]); del args[i:i+2]
rx = re.compile(args[0], re.I)
for f in args[1:]:
    t = open(f).read()
    sents = re.split(r"(?<=[.!?])\s+|\n", t)
    hits = [s.strip() for s in sents if rx.search(s)]
    print(f"=== {f} ({len(hits)})")
    for s in hits[:mx]:
        print(" -", s[:500])
