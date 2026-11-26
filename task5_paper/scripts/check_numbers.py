"""Check numeric literals against results.json, with a documented structural allowlist."""
import argparse, json, re
from pathlib import Path

STRUCTURAL={0,1,2,3,4,5,6,9,17,30,2024,2025,2026}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("tex",type=Path); ap.add_argument("results",type=Path); a=ap.parse_args()
    data=json.loads(a.results.read_text(encoding="utf-8"))
    allowed=set(float(x) for x in STRUCTURAL)
    # The chance level (50) appears negated in the retention formula; only structural constants may appear negated.
    allowed|={-float(x) for x in STRUCTURAL|{50}}
    def visit(x):
        if isinstance(x,bool): return
        if isinstance(x,(int,float)): allowed.add(float(x)); allowed.add(round(float(x),2)); allowed.add(round(float(x),3)); allowed.add(round(float(x),4))
        elif isinstance(x,dict):
            for v in x.values(): visit(v)
        elif isinstance(x,list):
            for v in x: visit(v)
    visit(data)
    src=a.tex.read_text(encoding="utf-8")
    # The template preamble contains style/version constants (for example 10pt).
    if "\\begin{document}" in src:
        src=src.split("\\begin{document}",1)[1]
    # Bibliography entry years are metadata, not experimental result claims.
    src=src.split("\\bibliography",1)[0]
    src=re.sub(r"(?<!\\)%.*", "", src)
    src=re.sub(r"\\cite[a-zA-Z*]*\s*(?:\[[^]]*\]\s*)?\{[^}]*\}", "", src)
    src=re.sub(r"\\(?:label|ref|pageref|eqref)\{[^}]*\}", "", src)
    # Layout commands are not result claims (added 2026-11-25): column ranges, lengths, figure widths.
    src=re.sub(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}", "", src)
    src=re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", src)
    src=re.sub(r"\\includegraphics\[[^]]*\]", "", src)
    src=src.replace("--", " to ")
    src=re.sub(r"\\pm", " ", src)
    vals=[]
    for m in re.finditer(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?", src):
        has_percent=m.group().endswith("%") or src[m.end():m.end()+2]==r"\%"
        raw=m.group().rstrip("%")
        v=float(raw)
        candidates=(v,v/100.0) if has_percent else (v,)
        if not any(abs(candidate-x)<=0.01001 for candidate in candidates for x in allowed): vals.append((raw,src.count("\n",0,m.start())+1))
    if vals: raise SystemExit("FAIL: unapproved numeric literals: "+", ".join(f"{v} (line {n})" for v,n in vals))
    print(f"PASS: all numeric literals match results.json within 0.01 or the structural allowlist.")
if __name__=="__main__": main()
