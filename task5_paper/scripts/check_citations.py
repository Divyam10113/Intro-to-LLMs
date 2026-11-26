"""Verify manuscript citation keys are pooled and present in references.bib."""
import argparse, json, re
from pathlib import Path

def keys_in_bib(text):
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", text))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("tex", type=Path); ap.add_argument("bib", type=Path)
    ap.add_argument("pool", type=Path); ap.add_argument("--pool-bib", type=Path)
    a=ap.parse_args()
    tex=re.sub(r"%.*", "", a.tex.read_text(encoding="utf-8"))
    cites=set()
    for value in re.findall(r"\\cite[a-zA-Z*]*\s*(?:\[[^]]*\]\s*)?\{([^}]+)\}", tex):
        cites.update(x.strip() for x in value.split(",") if x.strip())
    bib=keys_in_bib(a.bib.read_text(encoding="utf-8"))
    pool=json.loads(a.pool.read_text(encoding="utf-8"))
    pooled={p["citation_key"] for p in pool}
    if a.pool_bib:
        pooled |= keys_in_bib(a.pool_bib.read_text(encoding="utf-8"))
    errors=[]
    if cites-bib: errors.append("cited but absent from bib: "+", ".join(sorted(cites-bib)))
    if bib-pooled: errors.append("bib entries outside approved pool: "+", ".join(sorted(bib-pooled)))
    if not cites: errors.append("no citations found")
    if errors: raise SystemExit("FAIL: "+"; ".join(errors))
    print(f"PASS: {len(cites)} cited keys resolve; {len(bib)} bibliography entries are pooled.")
if __name__=="__main__": main()
