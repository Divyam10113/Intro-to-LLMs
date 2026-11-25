import json

transcript_path = r"C:\Users\Vismay VS\.gemini\antigravity-ide\brain\5d5461e9-b3b0-4b43-b276-a239df12aeba\.system_generated\logs\transcript.jsonl"

last_user_content = ""
with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data.get("type") == "USER_INPUT":
            last_user_content = data.get("content", "")

print("Length of last user message:", len(last_user_content))
if "Code snippet" in last_user_content:
    print("Found 'Code snippet' in last user message!")
    part = last_user_content.split("Code snippet")[1]
    csv_text = part.split("76 rows emitted")[0].strip()
    lines = csv_text.strip().split("\n")
    print(f"Number of lines extracted: {len(lines)}")
    print("Header:", lines[0][:80])
    print("First row:", lines[1][:80])
    print("Last row:", lines[-1][:80])

    # Save to prompted_reranking.csv
    out_paths = [
        r"c:\Users\Vismay VS\Desktop\Intro TO llms project\SurveyGen\prompted_reranking.csv",
        r"c:\Users\Vismay VS\Desktop\Intro TO llms project\SurveyGen\Paper implementation\prompted_reranking.csv"
    ]
    for p in out_paths:
        with open(p, "w", encoding="utf-8") as out_f:
            out_f.write(csv_text + "\n")
        print(f"Saved {len(lines)} lines to {p}")
