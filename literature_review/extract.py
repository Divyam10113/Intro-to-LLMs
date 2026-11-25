import pandas as pd

df = pd.read_csv('paper_after_reranking.csv')

# Filter for papers with an LLM score of 3 or above
selected = df[df['relevance_llm_1to5'] >= 3.0]

# Sort by composite_score in descending order (highest score first)
selected = selected.sort_values(by='composite_score', ascending=False)

out = f"# Top Papers (Score 3.0 and Above)\n\n"
out += f"Here are the {len(selected)} papers that received a relevance score of 3 or higher, ordered by their composite relevance score.\n\n"

for i, (idx, row) in enumerate(selected.iterrows(), 1):
    score = row.get('composite_score', 0)
    llm_score = row.get('relevance_llm_1to5', 0)
    out += f"## {i}. {row['title']}\n"
    out += f"**LLM Relevance Score:** {llm_score}/5 | **Composite Score:** {score:.4f}\n\n"
    out += f"**Abstract:**\n{row['abstract']}\n\n"
    out += "---\n\n"

with open('top_papers_score_3_and_above.md', 'w', encoding='utf-8') as f:
    f.write(out)

print(f"Done! Extracted {len(selected)} papers with score >= 3. Saved to top_papers_score_3_and_above.md")
