"""Shared config, paths, LLM client (with model fallback + disk cache) and text helpers."""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_PAPER_DIR = ROOT / "base_paper"
TASK1_DIR = ROOT / "DELIVERABLES" / "pipeline" / "gap_inputs"
DATA_DIR = HERE / "data"            # downloaded PDFs, corpus jsonl (gitignored)
CACHE_DIR = HERE / "cache"          # LLM response cache (gitignored)
OUT_DIR = HERE / "outputs"
for d in (DATA_DIR, CACHE_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

BASE_OPENALEX_ID = "W4402703054"    # MVFA, CVPR 2024 (arXiv:2403.12570)
BASE_TITLE = "Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images"

# Optional limitations/.env with KEY=value lines (gitignored), e.g. AGENTROUTER_API_KEY=...
if (HERE / ".env").exists():
    for _line in (HERE / ".env").read_text().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# -----------------------------------------------------------------------------
# Models (AgentRouter, OpenAI-compatible endpoint)
# -----------------------------------------------------------------------------
AGENTROUTER_BASE_URL = os.getenv("AGENTROUTER_BASE_URL", "https://agentrouter.org/v1")
# Generation agents: tried in order, falling back on any error.
GEN_MODELS = os.getenv("GEN_MODELS", "gpt-6-astra,gpt-6.1-sol").split(",")
# Evaluation (LLM-as-judge for coverage / F-S-I): a different model family first,
# so the system is not graded by the same model that generated it.
EVAL_MODELS = os.getenv("EVAL_MODELS", "gpt-6.1-sol,gpt-6-astra").split(",")

# NVIDIA NIM (OpenAI-compatible chat completions, key from build.nvidia.com)
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_GEN_MODELS = os.getenv("NIM_GEN_MODELS",
                           "minimaxai/minimax-m3.1-flash,z-ai/glm-5.3").split(",")
NIM_EVAL_MODELS = os.getenv("NIM_EVAL_MODELS",
                            "nvidia/nemotron-3-ultra-550b-a55b,z-ai/glm-5.3-flash").split(",")

# Which provider generates (LLM_PROVIDER) and which one judges in evaluate.py (EVAL_PROVIDER).
# Keep EVAL_PROVIDER fixed across all systems so their scores are comparable.
GEN_PROVIDER = os.getenv("LLM_PROVIDER", "agentrouter")
EVAL_PROVIDER = os.getenv("EVAL_PROVIDER", GEN_PROVIDER)
if GEN_PROVIDER == "nim":
    GEN_MODELS = NIM_GEN_MODELS
if EVAL_PROVIDER == "nim":
    EVAL_MODELS = NIM_EVAL_MODELS

AGENTROUTER_HEADERS = {
    "User-Agent": "opencode/1.0",
    "X-Stainless-Lang": "js",
    "X-Stainless-OS": "MacOS",
    "X-Stainless-Package-Version": "0.1.0",
    "X-Stainless-Runtime": "node",
    "X-Stainless-Retry-Count": "0",
}

_client = None
_nim_client = None


def _get_client(provider="agentrouter"):
    global _client, _nim_client
    if provider == "nim":
        if _nim_client is None:
            from openai import OpenAI
            key = os.getenv("NVIDIA_API_KEY")
            if not key:
                sys.exit("Set NVIDIA_API_KEY first (in limitations/.env)")
            _nim_client = OpenAI(api_key=key, base_url=NIM_BASE_URL, timeout=600, max_retries=0)
        return _nim_client
    if _client is None:
        from openai import OpenAI
        key = os.getenv("AGENTROUTER_API_KEY")
        if not key:
            sys.exit("Set AGENTROUTER_API_KEY first:  export AGENTROUTER_API_KEY=...")
        _client = OpenAI(
            api_key=key,
            base_url=AGENTROUTER_BASE_URL,
            timeout=600,
            max_retries=0,
            default_headers=AGENTROUTER_HEADERS,
        )
    return _client


MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "5"))   # per model, for dropped/overloaded connections
_UNAVAILABLE = set()                               # models the router has no channel for
_RESPONSES_API = {m for m in GEN_MODELS + EVAL_MODELS if m.startswith("gpt-")}  # need /v1/responses


def _call(client, model, system, user, temperature):
    """Streamed call: long generations otherwise sit silent until done and get cut by
    proxy idle timeouts (seen as APIConnectionError on every full-paper call)."""
    parts = []
    if model in _RESPONSES_API:
        kwargs = dict(model=model, instructions=system, input=user, stream=True)
        if temperature is not None:
            kwargs["temperature"] = temperature
        for event in client.responses.create(**kwargs):
            if event.type == "response.output_text.delta":
                parts.append(event.delta)
            elif event.type in ("response.failed", "error"):
                raise RuntimeError(f"stream error: {event}")
        return "".join(parts)
    kwargs = dict(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                  stream=True)
    if temperature is not None:
        kwargs["temperature"] = temperature
    for chunk in client.chat.completions.create(**kwargs):
        if chunk.choices and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
    return "".join(parts)


def _cache_path(models, system, user, temperature):
    h = hashlib.sha256(json.dumps([models, system, user, temperature]).encode()).hexdigest()[:24]
    return CACHE_DIR / f"{h}.json"


def chat(system, user, *, models=None, temperature=0.3, tag="llm", use_cache=True, provider=None):
    """Call the first model in `models` that works. Responses are cached on disk,
    so re-running the pipeline after a crash does not re-pay for finished calls."""
    models = models or GEN_MODELS
    # NIM model ids are "org/model"; AgentRouter ids have no slash
    provider = provider or ("nim" if all("/" in m for m in models) else "agentrouter")
    cp = _cache_path(models, system, user, temperature)
    if use_cache and cp.exists():
        return json.loads(cp.read_text())["content"]

    client = _get_client(provider)
    last_err = None
    for model in models:
        if model in _UNAVAILABLE:
            continue
        for attempt in range(MAX_ATTEMPTS):
            try:
                content = _call(client, model, system, user, temperature)
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()  # reasoning models
                if not content.strip():
                    raise RuntimeError("empty response")
                print(f"  [{tag}] {model} ok ({len(content)} chars)")
                cp.write_text(json.dumps({"model": model, "tag": tag, "content": content}))
                return content
            except Exception as e:  # noqa: BLE001 - any failure moves us down the chain
                last_err = e
                msg = str(e)
                # Some models reject a custom temperature; retry the same model without it.
                if "temperature" in msg.lower() and temperature is not None:
                    temperature = None
                    continue
                # Model only served via the Responses API -> switch API and retry immediately.
                if "/v1/responses" in msg and model not in _RESPONSES_API:
                    _RESPONSES_API.add(model)
                    continue
                # Router has no backend for this model at all -> skip it for the rest of the run.
                if "无可用渠道" in msg or "no available channel" in msg.lower() or "Error code: 404" in msg:
                    print(f"  [{tag}] {model} unavailable on the router; skipping it from now on")
                    _UNAVAILABLE.add(model)
                    break
                import openai
                import httpx
                transient = (isinstance(e, (openai.APIConnectionError, openai.RateLimitError,
                                            openai.InternalServerError, httpx.TransportError))
                             or any(s in msg for s in ("429", "500", "502", "503", "504", "timeout", "Timeout",
                                                   "server_error", "unavailable", "Unavailable", "overloaded",
                                                   "rate limit", "rate_limit")))
                print(f"  [{tag}] {model} failed (attempt {attempt + 1}): {type(e).__name__}: {msg[:160]}")
                if transient and attempt < MAX_ATTEMPTS - 1:
                    time.sleep(20 * (attempt + 1))
                    continue
                break
    raise RuntimeError(f"All models failed for {tag}: {last_err}")


def chat_json(system, user, **kw):
    """chat() that must return JSON. Re-asks once if the reply is not parseable."""
    out = chat(system, user, **kw)
    try:
        return parse_json(out)
    except ValueError:
        fix = chat("Return only valid JSON. No prose, no code fences.",
                   f"Convert this into the valid JSON it was meant to be:\n\n{out}",
                   models=kw.get("models"), temperature=0, tag=kw.get("tag", "llm") + "-fixjson")
        return parse_json(fix)


def parse_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    # try the outermost structure first: whichever of [ / { appears earliest
    pairs = sorted((("[", "]"), ("{", "}")), key=lambda p: text.find(p[0]) if p[0] in text else len(text))
    for opener, closer in pairs:
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError("no JSON found in model output")


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------
def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def title_key(s):
    return norm(s)[:120]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_base_paper():
    """Return [(section_name, text)] for the base paper, read from its LaTeX sources."""
    order = ["0_abstract", "1_intro_new", "2_related", "3_method", "4_experiment", "5_conclusion", "X_supp"]
    names = {"0_abstract": "Abstract", "1_intro_new": "Introduction", "2_related": "Related Work",
             "3_method": "Method", "4_experiment": "Experiments", "5_conclusion": "Conclusion",
             "X_supp": "Supplementary"}
    out = []
    for stem in order:
        p = BASE_PAPER_DIR / "sec" / f"{stem}.tex"
        if p.exists():
            out.append((names[stem], strip_latex(p.read_text(encoding="utf-8", errors="ignore"))))
    return out


def strip_latex(t):
    t = re.sub(r"(?<!\\)%.*", "", t)                                  # comments
    t = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", lambda m: _captions(m.group(0)), t, flags=re.S)
    t = re.sub(r"\\(cite|citep|citet|ref|eqref|label)\{[^}]*\}", "", t)
    t = re.sub(r"\\(textbf|textit|emph|underline|mathrm|text)\{([^}]*)\}", r"\2", t)
    t = re.sub(r"\\(section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}", r"\n\n## \2\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _captions(fig):
    caps = re.findall(r"\\caption\{(.*?)\}\s*(?:\\label|\n|$)", fig, re.S)
    return "\n".join(f"[Figure caption] {c}" for c in caps)


def base_paper_text(max_chars=None):
    skip = {"Supplementary"} if os.getenv("NO_SUPP") else set()
    txt = "\n\n".join(f"# {name}\n{body}" for name, body in load_base_paper() if name not in skip)
    return txt[:max_chars] if max_chars else txt
