"""
answer.py — helper for the file backend: write answers to pending prompts, addressed by their tag.

    python answer.py list [tag-prefix]          # show pending tags
    python answer.py show <tag>                 # print one pending prompt
    python answer.py write answers.json         # {"<tag>": "<response text>", ...} -> .response.txt files
"""
import json
import sys
from pathlib import Path

PENDING = Path(__file__).resolve().parent / "logs" / "pending"


def pending():
    for p in sorted(PENDING.glob("*.json")):
        r = json.loads(p.read_text())
        yield p.stem, r


def main():
    cmd = sys.argv[1]
    if cmd == "list":
        pre = sys.argv[2] if len(sys.argv) > 2 else ""
        for k, r in sorted(pending(), key=lambda kr: kr[1]["tag"]):
            if r["tag"].startswith(pre):
                print(k, r["tag"])
    elif cmd == "show":
        for k, r in pending():
            if r["tag"] == sys.argv[2]:
                print(r["system"], "\n---\n", r["user"])
    elif cmd == "write":
        answers = json.loads(Path(sys.argv[2]).read_text())
        by_tag = {r["tag"]: k for k, r in pending()}
        n = 0
        for tag, text in answers.items():
            if tag not in by_tag:
                print("not pending:", tag)
                continue
            (PENDING / f"{by_tag[tag]}.response.txt").write_text(text)
            n += 1
        print(f"wrote {n} answer(s); {len(by_tag) - n} still pending")


if __name__ == "__main__":
    main()
