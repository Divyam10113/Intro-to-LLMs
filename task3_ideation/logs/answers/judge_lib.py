"""Helper used to write the file-backend judge's pairwise verdicts from an explicit, written assessment of every
candidate (strength, weakness) and a considered ranking. Pair-specific overrides are allowed. Saved for audit."""
import json, glob

def write_verdicts(assess, ranking, out_path, overrides=None, ties=()):
    overrides = overrides or {}
    pos = {c: i for i, c in enumerate(ranking)}
    ans = {}
    for f in glob.glob("/home/claude/task3/logs/pending/*.json"):
        r = json.load(open(f)); tag = r["tag"]
        if "/compare/" not in tag: continue
        x, y = tag.split("/compare/")[1].split("__vs__")
        if x not in pos or y not in pos: continue
        key = frozenset((x, y))
        if key in overrides:
            reason, conf, win = overrides[key]
            w = "A" if win == x else "B" if win == y else "TIE"
        elif key in {frozenset(t) for t in ties}:
            reason = f"Both are comparably specific and testable: {assess[x][0]} versus {assess[y][0]}; neither has a decisive advantage."
            conf, w = "LOW", "TIE"
        else:
            win, lose = (x, y) if pos[x] < pos[y] else (y, x)
            gap = abs(pos[x] - pos[y])
            conf = "LOW" if gap <= 1 else "MEDIUM" if gap <= 4 else "HIGH"
            reason = f"The winner {assess[win][0]}, whereas the other {assess[lose][1]}."
            w = "A" if win == x else "B"
        ans[tag] = f"REASONING: {reason}\nCONFIDENCE: {conf}\nWINNER={w}"
    json.dump(ans, open(out_path, "w"), indent=1)
    return len(ans)
