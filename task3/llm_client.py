"""
llm_client.py — one place where every LLM call in the Task 3 pipeline goes through.

Why this file exists
--------------------
All three methods we implement (ResearchAgent, HypER, HypoEvolve) are LLM pipelines:
every "agent" is just a prompt template sent to a language model. This module makes the
model swappable without touching the pipeline code, and it records every prompt/response
so that any number in the final report can be traced back to the exact call that produced it.

Backends
--------
* "openai" : any OpenAI-compatible endpoint (Groq, AgentRouter, OpenAI, a local vLLM, ...).
             Configure with LLM_BASE_URL, LLM_MODEL and the API key in LLM_API_KEY.
* "file"   : the prompt is written to logs/pending/<key>.json and the run stops.
             Someone (in our case Claude, acting as the model) writes the answer to
             logs/pending/<key>.response.txt, `python llm_client.py ingest` moves it into the
             cache, and the stage is re-run. Because every call is cached by a hash of its
             inputs and all randomness is seeded, re-running a stage replays the cached
             calls and continues exactly where it stopped.

Roles
-----
Calls are made with a role: "generator" (writes problems / ideas / offspring) or "judge"
(reviews, pairwise comparisons, link scoring). The judge can be pointed at a *different*
model with JUDGE_BACKEND / JUDGE_MODEL / JUDGE_BASE_URL / JUDGE_API_KEY. This is how the
"hybrid" plan works: re-running only the judging steps with e.g. a Groq-hosted Llama-70B
gives an independent check on the rankings (see README, "Re-judging with Groq").

Example (Groq):
    export LLM_BACKEND=openai
    export LLM_BASE_URL=https://api.groq.com/openai/v1
    export LLM_API_KEY=gsk_...
    export LLM_MODEL=llama-3.3-70b-versatile
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "logs" / "cache"
PENDING_DIR = ROOT / "logs" / "pending"
CALL_LOG = ROOT / "logs" / "calls.jsonl"

# Name recorded for calls answered through the file backend in this run.
FILE_BACKEND_MODEL = os.environ.get("FILE_BACKEND_MODEL", "claude-opus-5-5 (file backend)")


class PendingCalls(Exception):
    """Raised in file mode when some prompts have no cached answer yet."""

    def __init__(self, keys):
        self.keys = keys
        super().__init__(
            f"{len(keys)} LLM call(s) waiting for an answer in {PENDING_DIR}. "
            "Write <key>.response.txt files, run `python llm_client.py ingest`, then re-run this stage."
        )


@dataclass
class Config:
    backend: str
    model: str
    base_url: str | None
    api_key: str | None

    @classmethod
    def from_env(cls, role: str) -> "Config":
        def get(name, default=None):
            if role == "judge" and os.environ.get(f"JUDGE_{name}"):
                return os.environ[f"JUDGE_{name}"]
            return os.environ.get(f"LLM_{name}", default)

        backend = get("BACKEND", "file")
        model = get("MODEL", FILE_BACKEND_MODEL if backend == "file" else "llama-3.3-70b-versatile")
        return cls(backend=backend, model=model, base_url=get("BASE_URL"), api_key=get("API_KEY"))


def _key(model: str, system: str, user: str, sample: int) -> str:
    h = hashlib.sha256(json.dumps([model, system, user, sample]).encode()).hexdigest()
    return h[:16]


class LLM:
    def __init__(self, role: str = "generator"):
        self.role = role
        self.cfg = Config.from_env(role)
        self._client = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        PENDING_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public
    def call(self, system: str, user: str, tag: str, temperature: float = 0.7, sample: int = 0) -> str:
        return self.batch([dict(system=system, user=user, tag=tag, temperature=temperature, sample=sample)])[0]

    def batch(self, requests: list[dict]) -> list[str]:
        """Answer a list of independent requests. In file mode, all missing ones are
        written to the pending folder together (so they can be answered in one sitting)."""
        out, missing = [], []
        for r in requests:
            k = _key(self.cfg.model, r["system"], r["user"], r.get("sample", 0))
            cached = self._read_cache(k)
            if cached is not None:
                out.append(cached)
                continue
            if self.cfg.backend == "file":
                self._write_pending(k, r)
                missing.append(k)
                out.append(None)
            else:
                text = self._call_api(r)
                self._write_cache(k, r, text)
                out.append(text)
        if missing:
            raise PendingCalls(missing)
        return out

    # ----------------------------------------------------------------- internals
    def _read_cache(self, k):
        p = CACHE_DIR / f"{k}.json"
        if p.exists():
            return json.loads(p.read_text())["response"]
        return None

    def _write_cache(self, k, r, text):
        rec = dict(key=k, role=self.role, model=self.cfg.model, backend=self.cfg.backend, tag=r["tag"],
                   temperature=r.get("temperature"), sample=r.get("sample", 0),
                   system=r["system"], user=r["user"], response=text, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        (CACHE_DIR / f"{k}.json").write_text(json.dumps(rec, indent=1))
        with open(CALL_LOG, "a") as f:
            f.write(json.dumps({k2: rec[k2] for k2 in ("key", "role", "model", "tag", "time")}) + "\n")

    def _write_pending(self, k, r):
        p = PENDING_DIR / f"{k}.json"
        if not p.exists():
            p.write_text(json.dumps(dict(key=k, role=self.role, model=self.cfg.model, **r), indent=1))

    def _call_api(self, r, retries: int = 6):
        if self._client is None:
            from openai import OpenAI  # imported lazily so file mode needs no extra install
            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
        delay = 5
        for attempt in range(retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=r.get("temperature", 0.7),
                    messages=[{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                )
                return resp.choices[0].message.content
            except Exception as e:  # rate limits etc. -> exponential backoff
                if attempt == retries - 1:
                    raise
                print(f"[llm_client] {type(e).__name__}: {e} -- retrying in {delay}s", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 120)


def ingest() -> int:
    """Move answered pending prompts into the cache."""
    n = 0
    for resp_file in sorted(PENDING_DIR.glob("*.response.txt")):
        k = resp_file.name.split(".")[0]
        req_file = PENDING_DIR / f"{k}.json"
        if not req_file.exists():
            continue
        r = json.loads(req_file.read_text())
        llm = LLM(r.get("role", "generator"))
        llm._write_cache(k, r, resp_file.read_text().strip())
        req_file.unlink()
        resp_file.unlink()
        n += 1
    return n


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        print(f"ingested {ingest()} answer(s)")
    elif len(sys.argv) > 1 and sys.argv[1] == "pending":
        for p in sorted(PENDING_DIR.glob("*.json")):
            r = json.loads(p.read_text())
            print(p.stem, r["tag"])
    else:
        print(__doc__)
