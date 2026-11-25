import csv
import json

# 1. Check CSV
with open("prompted_reranking.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
selected = [r for r in rows if r["tag"] == "selected"]
print(f"CSV: Total rows = {len(rows)}, Selected = {len(selected)}, Columns = {len(rows[0])}")

# 2. Check Prompted_ref.json
with open("Prompted_ref.json", "r", encoding="utf-8") as f:
    p_data = json.load(f)
refs = p_data["references"]
print(f"Prompted_ref.json: Retained references = {len(refs)}")
print(f"Prompted_ref.json: Score 5 = {sum(1 for r in refs if r['relevance_llm_1to5']==5)}")
print(f"Prompted_ref.json: Score 4 = {sum(1 for r in refs if r['relevance_llm_1to5']==4)}")
print(f"Prompted_ref.json: Score 3 = {sum(1 for r in refs if r['relevance_llm_1to5']==3)}")
print(f"Prompted_ref.json: Score 2 = {sum(1 for r in refs if r['relevance_llm_1to5']==2)}")

# 3. Check ref.json
with open("ref.json", "r", encoding="utf-8") as f:
    r_data = json.load(f)
print(f"ref.json: Total references = {len(r_data)}")

# 4. Check Tex file
with open("literature_review_2page.tex", "r", encoding="utf-8") as f:
    tex = f.read()
print(f"LaTeX: Length = {len(tex)} chars, twoside={'twoside' in tex}")
