"""Compile with Tectonic and verify 4 content pages plus one references page."""
import argparse, os, re, subprocess
from pathlib import Path
import pymupdf

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("tex",type=Path); ap.add_argument("--tectonic",type=Path,required=True); ap.add_argument("--references-only-page",type=int,default=5); a=ap.parse_args()
    tex=a.tex.resolve(); work=tex.parent
    run=subprocess.run([str(a.tectonic.resolve()),"--keep-logs","--keep-intermediates","--reruns","2",tex.name],cwd=work,capture_output=True,text=True)
    pdf=work/(tex.stem+".pdf")
    if run.returncode or not pdf.exists(): raise SystemExit("FAIL: Tectonic compilation failed\n"+run.stdout[-3000:]+run.stderr[-3000:])
    doc=pymupdf.open(pdf)
    if len(doc)!=5: raise SystemExit(f"FAIL: expected exactly 5 pages; found {len(doc)}")
    pages=[p.get_text() for p in doc]
    for i,text in enumerate(pages[:4],1):
        if re.search(r"(?im)^\s*(references|bibliography)\s*$",text): raise SystemExit(f"FAIL: bibliography heading appears on content page {i}")
    if not re.search(r"(?i)references|bibliography",pages[4]): raise SystemExit("FAIL: page 5 does not appear to contain references")
    if re.search(r"(?i)appendix", "\n".join(pages)): raise SystemExit("FAIL: appendix text found")
    log=work/(tex.stem+".log")
    if log.exists():
        l=log.read_text(errors="ignore")
        if re.search(r"Undefined (?:control sequence|references)|Citation .* undefined",l,re.I): raise SystemExit("FAIL: undefined references/citations in TeX log")
    print(f"PASS: {pdf} has 4 content pages and a references-only page 5.")
if __name__=="__main__": main()
