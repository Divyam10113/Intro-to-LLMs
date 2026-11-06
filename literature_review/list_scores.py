import pandas as pd
import json
import re

log_path = r"C:\Users\Vismay VS\.gemini\antigravity-ide\brain\56516b08-5860-4f4b-b00f-b590527413d4\.system_generated\tasks\task-469.log"
with open(log_path, 'r', encoding='utf-8') as f:
    log_data = f.read()

scores = {}
for match in re.finditer(r'\[(\d+)/333\] Score=(\d)', log_data):
    scores[int(match.group(1)) - 1] = int(match.group(2))

with open('final_ref_set.json', 'r', encoding='utf-8') as f:
    df = pd.DataFrame(json.load(f))
    
df = df[df['abstract'].apply(lambda x: isinstance(x, str) and x.strip() != '')].reset_index(drop=True)

print(f"Total graded so far: {len(scores)}")
for idx, score in scores.items():
    if idx < len(df):
        print(f"Score: {score} | Title: {df.iloc[idx]['title']}")
