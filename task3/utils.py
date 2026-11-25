"""Small shared helpers."""
import json
import re


def parse_json(text: str):
    """Extract the first JSON object/array from an LLM response (tolerates ```json fences and prose)."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1])
    depth, in_str, esc = 0, False, False
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("no complete JSON found in response")


def parse_fields(text: str, fields: list[str]) -> dict:
    """Parse 'Field: value' blocks such as ResearchAgent's 'Problem: ... Rationale: ...' format."""
    out = {}
    pattern = "|".join(re.escape(f) for f in fields)
    parts = re.split(rf"^\s*\**\s*({pattern})\s*\**\s*:\s*", text, flags=re.M | re.I)
    for i in range(1, len(parts) - 1, 2):
        key = next(f for f in fields if f.lower() == parts[i].lower())
        out[key] = parts[i + 1].strip()
    return out
