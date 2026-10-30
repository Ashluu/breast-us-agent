"""Search Europe PMC (OA only) and print pmcid, year, journal, license, cites, title."""
import json, sys, subprocess, urllib.parse
q = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
sort = sys.argv[3] if len(sys.argv) > 3 else "CITED desc"
url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" +
       urllib.parse.urlencode({"query": q + " AND OPEN_ACCESS:y", "format": "json",
                               "resultType": "core", "pageSize": n, "sort": sort}))
d = json.loads(subprocess.run(["curl", "-s", "--max-time", "60", url], capture_output=True, check=True).stdout)
print("hits", d["hitCount"])
for r in d["resultList"]["result"]:
    j = r.get("journalInfo", {}).get("journal", {}).get("medlineAbbreviation") or r.get("journalInfo", {}).get("journal", {}).get("title")
    print(r.get("pmcid"), r.get("pubYear"), j, r.get("license"), r.get("citedByCount"), "|", (r.get("title") or "")[:120])
