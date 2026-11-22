"""Export the Claude Code chat transcripts of Task 4 to one Markdown file.

    python export_transcript.py OUT.md MAIN1.jsonl [MAIN2.jsonl ...] [--side SIDE.jsonl]

Each MAIN file is rendered in order as one part; a --side file (a forked side
conversation) is rendered as an appendix with only the messages that are not
already in the main files.

Included: every user message, every assistant reply, every tool call (tool
name, its description and input) and tool output (truncated to keep the file
readable). Left out: the model's hidden reasoning blocks, and system-inserted
context (<system-reminder> blocks), which is app plumbing, not conversation.
The user's e-mail address is redacted.
"""
import json
import re
import sys

MAX_IN, MAX_OUT = 4000, 2500
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
KEEP_EMAIL = {"noreply@anthropic.com"}


def _author_local_part():
    """Local part of the repo's git author e-mail (read at run time, never stored here)."""
    import subprocess
    try:
        e = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None
    return re.escape(e.split("@")[0]) if "@" in e else None


_LOCAL = _author_local_part()


def redact(s):
    s = EMAIL.sub(lambda m: m.group(0) if m.group(0) in KEEP_EMAIL else "[email redacted]", s)
    if _LOCAL:
        s = re.sub(_LOCAL[:8] + r"\w*", "[email redacted]", s, flags=re.I)  # also partial mentions
    return s


def strip_sys(s):
    return re.sub(r"<system-reminder>.*?</system-reminder>", "", s, flags=re.S).strip()


def trunc(s, n):
    return s if len(s) <= n else s[:n] + f"\n… [{len(s) - n} more characters truncated]"


def fence(s):
    s = s.replace("```", "ˋˋˋ")
    return f"```\n{s}\n```"


def load(path):
    out = []
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") in ("user", "assistant") and d.get("uuid"):
            out.append(d)
        elif d.get("type") == "attachment" and d.get("uuid"):
            a = d.get("attachment") or {}
            # a message the user typed while Claude was mid-task
            if a.get("type") == "queued_command" and (a.get("origin") or {}).get("kind") == "human":
                out.append({"type": "queued", "uuid": d["uuid"], "timestamp": d.get("timestamp", ""),
                            "text": str(a.get("prompt", ""))})
    return out


def tool_result_text(c):
    x = c.get("content")
    if isinstance(x, list):
        parts = []
        for y in x:
            if y.get("type") == "text":
                parts.append(y["text"])
            elif y.get("type") == "image":
                parts.append("[image]")
        x = "\n".join(parts)
    return strip_sys(str(x or ""))


def render(d):
    ts = d.get("timestamp", "")[:19].replace("T", " ")
    msg = d.get("message", {}) or {}
    content = msg.get("content")
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
    lines = []
    if d["type"] == "queued":
        return redact(f"### 🧑 User (sent while Claude was working) · {ts}\n\n{d['text'].strip()}")
    if d["type"] == "user":
        if d.get("isMeta"):
            return ""
        for b in blocks:
            if b.get("type") == "text":
                t = strip_sys(b["text"])
                if not t:
                    continue
                if t.startswith("<task-notification>"):
                    s = re.search(r"<summary>(.*?)</summary>", t, re.S)
                    lines.append(f"> _[background task notification: {s.group(1).strip() if s else 'event'}]_")
                elif d.get("isCompactSummary") or t.startswith("This session is being continued"):
                    lines.append(f"**[{ts}] Automatic context summary (inserted by the app when the "
                                 f"previous context filled up):**\n\n<details><summary>show summary</summary>\n\n"
                                 f"{t}\n\n</details>")
                else:
                    lines.append(f"### 🧑 User · {ts}\n\n{t}")
            elif b.get("type") == "tool_result":
                t = tool_result_text(b)
                if t:
                    lines.append(f"<details><summary>tool output</summary>\n\n{fence(trunc(t, MAX_OUT))}\n\n</details>")
            elif b.get("type") == "image":
                lines.append("_[image attached]_")
    else:
        for b in blocks:
            if b.get("type") == "text" and b["text"].strip():
                lines.append(f"### 🤖 Claude · {ts}\n\n{b['text'].strip()}")
            elif b.get("type") == "tool_use":
                inp = dict(b.get("input") or {})
                desc = inp.pop("description", "")
                head = f"**🔧 {b.get('name')}**" + (f" — {desc}" if desc else "")
                body = json.dumps(inp, ensure_ascii=False, indent=1)
                if "command" in inp and len(inp) <= 3:
                    body = inp["command"]
                lines.append(f"{head}\n\n<details><summary>input</summary>\n\n{fence(trunc(body, MAX_IN))}\n\n</details>")
    return redact("\n\n".join(lines))


def main():
    args = sys.argv[1:]
    out = args.pop(0)
    side = None
    if "--side" in args:
        i = args.index("--side")
        side = args[i + 1]
        del args[i:i + 2]
    doc = ["# Task 4: full chat transcript\n",
           "Conversation between the author and Claude (Claude Code, Opus 5.5) that produced Task 4, "
           "exported by `task4/export_transcript.py` from the app's session files. "
           "Times are UTC. Tool outputs are truncated; the model's hidden reasoning and "
           "app-inserted system context are left out; the author's e-mail is redacted.\n"]
    seen = set()
    for n, path in enumerate(args, 1):
        ms = load(path)
        doc.append(f"\n---\n\n## Part {n}: {ms[0]['timestamp'][:16].replace('T', ' ')} → "
                   f"{ms[-1]['timestamp'][:16].replace('T', ' ')} UTC\n")
        for d in ms:
            seen.add(d["uuid"])
            r = render(d)
            if r:
                doc.append(r)
    if side:
        ms = [d for d in load(side) if d["uuid"] not in seen]
        doc.append("\n---\n\n## Appendix: side conversation forked from Part 1\n")
        for d in ms:
            r = render(d)
            if r:
                doc.append(r)
    open(out, "w").write("\n\n".join(doc) + "\n")
    print(out, sum(len(x) for x in doc), "chars")


if __name__ == "__main__":
    main()
