"""Shared utilities: logging, JSON I/O, HTTP with polite retries, and a cached
Groq LLM client.

The LLM client is deliberately provider-agnostic (any OpenAI-compatible
endpoint).  Two behaviours matter for reproducibility and cost:

* every request/response pair is cached on disk keyed by a hash of the exact
  payload, so re-running the pipeline is free and deterministic;
* every response - including raw text and token usage - is appended to a JSONL
  audit log, which is what the ``IMPLEMENTATION_NOTES.md`` file quotes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests

from . import config

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured: set = set()


def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Console logger plus a per-stage log file under ``intermediates/logs``."""
    logger = logging.getLogger(name)
    if name in _configured:
        return logger
    config.ensure_dirs()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter(_LOG_FORMAT)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    target = log_file or (config.LOG_DIR / f"{name}.log")
    fh = logging.FileHandler(target, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    _configured.add(name)
    return logger


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)
    tmp.replace(path)


def append_jsonl(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Any]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Text / DOI helpers
# --------------------------------------------------------------------------- #
def norm_doi(doi: Optional[str]) -> str:
    if not doi:
        return ""
    return str(doi).lower().replace("https://doi.org/", "").replace("http://doi.org/", "").strip()


def openalex_short_id(url_or_id: Optional[str]) -> str:
    if not url_or_id:
        return ""
    return str(url_or_id).rstrip("/").split("/")[-1]


def invert_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Reconstruct an abstract from an OpenAlex ``abstract_inverted_index``."""
    if not inverted_index:
        return ""
    positions: Dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return ""
    return " ".join(positions.get(i, "") for i in range(max(positions) + 1)).strip()


def normalize_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s.strip(" .,:;!?\"'()[]{}")


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[\\/*?\"<>|:\n\r\t]", "_", text)
    return text.strip()[:100] or "untitled"


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def clean_json_string(s: str) -> str:
    """Strip markdown code fences that LLMs like to add around JSON."""
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_first_int(text: str, lo: int, hi: int) -> Optional[int]:
    if not isinstance(text, str):
        return None
    m = re.search(rf"\b([{lo}-{hi}])\b", text)
    if not m:
        m = re.search(rf"[{lo}-{hi}]", text)
    return int(m.group(0)) if m else None


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class HttpClient:
    """requests wrapper with a global minimum interval and retry/backoff.

    Honours ``Retry-After`` on HTTP 429 and backs off exponentially on 5xx or
    transport errors, which is what the free OpenAlex/Semantic Scholar tiers
    require.
    """

    def __init__(
        self,
        min_interval: float = config.HTTP_MIN_INTERVAL,
        max_retries: int = config.HTTP_MAX_RETRIES,
        timeout: int = config.HTTP_TIMEOUT,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": f"QUAL-SG/1.0 (mailto:{config.OPENALEX_MAILTO})"}
        )
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.logger = logger or get_logger("http")
        self._lock = threading.Lock()
        self._last_call = 0.0
        self.n_requests = 0
        self.n_retries = 0
        #: Set when the provider answers 429 with a Retry-After larger than this
        #: many seconds (OpenAlex returns ~27 000 s once the daily budget is spent).
        #: Sleeping that long would hang the pipeline, so the source is disabled for
        #: the rest of the run and a warning is logged; callers see ``None``.
        self.blocked_until = 0.0
        self.retry_after_limit = float(os.environ.get("QUAL_SG_RETRY_AFTER_LIMIT", "120"))
        self._openalex_key = config.openalex_api_key()

    def _auth_headers(self, url: str) -> Optional[Dict[str, str]]:
        """OpenAlex key as a header, so it never appears in a logged URL."""
        if self._openalex_key and urlparse(url).netloc == "api.openalex.org":
            return {"Authorization": f"Bearer {self._openalex_key}"}
        return None

    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    # -- on-disk response cache (successful responses only) ----------------- #
    @staticmethod
    def _cache_path(kind: str, url: str, params: Optional[Dict[str, Any]]) -> Path:
        key = sha256_hex(json.dumps({"k": kind, "u": url, "p": params or {}}, sort_keys=True, default=str))
        return config.HTTP_CACHE_DIR / key[:2] / f"{key}.json"

    def _cache_get(self, kind: str, url: str, params: Optional[Dict[str, Any]]) -> Any:
        if not config.HTTP_CACHE:
            return None
        path = self._cache_path(kind, url, params)
        if not path.exists():
            return None
        try:
            self.n_cache_hits = getattr(self, "n_cache_hits", 0) + 1
            return json.loads(path.read_text(encoding="utf-8"))["body"]
        except Exception:  # noqa: BLE001 - a corrupt cache file is simply refetched
            return None

    def _cache_put(self, kind: str, url: str, params: Optional[Dict[str, Any]], body: Any) -> None:
        if not config.HTTP_CACHE:
            return
        path = self._cache_path(kind, url, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"url": url, "params": params, "ts": time.time(), "body": body},
                                   ensure_ascii=False), encoding="utf-8")

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        cached = self._cache_get("json", url, params)
        if cached is not None:
            return cached
        if time.time() < self.blocked_until:
            return None
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                self.n_requests += 1
                resp = self.session.get(url, params=params, headers=self._auth_headers(url), timeout=self.timeout)
                if resp.status_code == 200:
                    body = resp.json()
                    self._cache_put("json", url, params, body)
                    return body
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "5") or 5)
                    if retry_after > self.retry_after_limit:
                        self.blocked_until = time.time() + retry_after
                        self.logger.error(
                            "429 with Retry-After=%.0fs on %s -> source disabled for this run "
                            "(set QUAL_SG_RETRY_AFTER_LIMIT to wait instead)",
                            retry_after, url.split("?")[0],
                        )
                        return None
                    self.logger.warning("429 on %s -> sleeping %.1fs", url, retry_after)
                    time.sleep(min(retry_after, 60))
                elif resp.status_code in (400, 403, 404):
                    self.logger.error("HTTP %s on %s: %s", resp.status_code, url, resp.text[:200])
                    return None
                else:
                    self.logger.warning("HTTP %s on %s (attempt %s)", resp.status_code, url, attempt)
            except Exception as exc:  # noqa: BLE001 - transport errors are retryable
                last_err = exc
                self.logger.warning("Transport error on %s: %s", url, exc)
            self.n_retries += 1
            time.sleep(min(2.0 * attempt, 20))
        if last_err:
            self.logger.error("Giving up on %s: %s", url, last_err)
        return None

    def get_text(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        cached = self._cache_get("text", url, params)
        if cached is not None:
            return cached
        if time.time() < self.blocked_until:
            return None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, headers=self._auth_headers(url), timeout=self.timeout)
                if resp.status_code == 200:
                    self._cache_put("text", url, params, resp.text)
                    return resp.text
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "5") or 5)
                    if retry_after > self.retry_after_limit:
                        self.blocked_until = time.time() + retry_after
                        self.logger.error("429 with Retry-After=%.0fs -> source disabled for this run", retry_after)
                        return None
                    time.sleep(min(retry_after, 60))
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Transport error on %s: %s", url, exc)
            time.sleep(min(2.0 * attempt, 20))
        return None


# --------------------------------------------------------------------------- #
# LLM client
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Sliding-window limiter: at most ``rpm`` starts per 60 s."""

    def __init__(self, rpm: int) -> None:
        self.rpm = max(1, rpm)
        self._times: List[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                self._times = [t for t in self._times if now - t < 60.0]
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                sleep_for = 60.0 - (now - self._times[0]) + 0.05
            time.sleep(max(0.05, sleep_for))


@dataclass
class LLMResult:
    text: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency: float = 0.0
    cached: bool = False
    attempts: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


class PendingLLM(LLMError):
    """Raised by the file backend when a prompt is waiting for an answer.

    Steps must let it propagate (never swallow it as an ordinary failure): the
    run driver stops, the pending prompts are answered, and the step is re-run,
    which then replays every answered call from the cache.
    """


class LLMClient:
    """OpenAI-compatible chat client with disk cache, rate limiting and audit log."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = config.LLM_TEMPERATURE,
        cache_path: Optional[Path] = None,
        audit_path: Optional[Path] = None,
        rpm: Optional[int] = None,
        concurrency: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        fallbacks: Optional[Sequence[str]] = None,
    ) -> None:
        from openai import OpenAI  # imported lazily so steps 1-2 need no API key

        self.model = model
        self.fallbacks = list(fallbacks if fallbacks is not None else config.MODEL_FALLBACKS)
        self.temperature = temperature
        self.logger = logger or get_logger("llm")
        key = api_key if api_key is not None else config.api_key()
        if not key:
            raise LLMError(
                f"No API key. Set {config.LLM_API_KEY_ENV} (see qual_sg/.env.example)."
            )
        self.client = OpenAI(base_url=base_url or config.LLM_BASE_URL, api_key=key, timeout=config.LLM_REQUEST_TIMEOUT)
        self.concurrency = concurrency or config.LLM_CONCURRENCY
        self.limiter = RateLimiter(rpm if rpm is not None else int(os.environ.get("QUAL_SG_RPM", "28")))
        self.cache_path = Path(cache_path) if cache_path else (config.CACHE_DIR / "llm_cache.jsonl")
        self.audit_path = Path(audit_path) if audit_path else (config.CACHE_DIR / "llm_audit.jsonl")
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load_cache()
        self.usage_totals: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_failures = 0

    # -- cache ------------------------------------------------------------- #
    def _load_cache(self) -> None:
        for row in read_jsonl(self.cache_path):
            if "key" in row:
                self._cache[row["key"]] = row
        self.logger.info("LLM cache: %d entries (%s)", len(self._cache), self.cache_path)

    @staticmethod
    def _key(model: str, messages: List[Dict[str, str]], temperature: float, json_mode: bool) -> str:
        payload = json.dumps(
            {"m": model, "msgs": messages, "t": temperature, "j": json_mode},
            ensure_ascii=False,
            sort_keys=True,
        )
        return sha256_hex(payload)

    # -- calls ------------------------------------------------------------- #
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        use_cache: bool = True,
        tag: str = "",
    ) -> LLMResult:
        model = model or self.model
        temperature = self.temperature if temperature is None else temperature
        key = self._key(model, messages, temperature, json_mode)
        exp_dir = "cache" if use_cache else "cache"
        exp = config.CACHE_DIR / f"{exp_dir}_{key[:16]}.json"
        with self._lock:
            if use_cache and key in self._cache:
                self.n_cache_hits += 1
                row = self._cache[key]
                return LLMResult(row["text"], row["model"], row.get("usage", {}), 0.0, True, 1, row)
            if not use_cache and exp.exists():
                pass
        if use_cache and exp.exists():
            try:
                row = read_json(exp)
                with self._lock:
                    self._cache[key] = row
                self.n_cache_hits += 1
                return LLMResult(row["text"], row["model"], row.get("usage", {}), 0.0, True, 1, row)
            except Exception:  # noqa: BLE001
                pass

        models = [model] + [m for m in self.fallbacks if m != model]
        last_err: Optional[Exception] = None
        for m in models:
            for attempt in range(1, config.LLM_MAX_RETRIES + 1):
                try:
                    self.limiter.acquire()
                    t0 = time.time()
                    kwargs: Dict[str, Any] = dict(
                        model=m,
                        messages=messages,
                        temperature=temperature,
                    )
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    if max_tokens:
                        kwargs["max_tokens"] = max_tokens
                    if reasoning_effort:
                        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
                    resp = self.client.chat.completions.create(**kwargs)
                    latency = time.time() - t0
                    text = (resp.choices[0].message.content or "").strip()
                    usage = {}
                    if getattr(resp, "usage", None) is not None:
                        usage = {
                            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                            "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                            "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
                        }
                    row = {
                        "key": key,
                        "model": m,
                        "text": text,
                        "usage": usage,
                        "tag": tag,
                        "ts": time.time(),
                    }
                    self._store(key, row, exp, audit=True)
                    with self._lock:
                        self.n_calls += 1
                        for k, v in usage.items():
                            self.usage_totals[k] = self.usage_totals.get(k, 0) + v
                    return LLMResult(text, m, usage, latency, False, attempt, row)
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    msg = str(exc)
                    status = getattr(exc, "status_code", None)
                    self.logger.warning(
                        "LLM call failed (model=%s attempt=%d status=%s): %s",
                        m, attempt, status, msg[:200],
                    )
                    if status in (400, 404) and "model" in msg.lower():
                        break  # try next model
                    sleep_for = min(3.0 * attempt, 30.0)
                    retry_after = None
                    m_retry = re.search(r"retry[_ ]?after['\"]?[:= ]+(\d+(?:\.\d+)?)", msg, re.IGNORECASE)
                    if m_retry:
                        retry_after = float(m_retry.group(1))
                    time.sleep(retry_after if retry_after else sleep_for)
        self.n_failures += 1
        raise LLMError(f"All models failed for tag={tag!r}: {last_err}")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        tag: str = "",
    ) -> Tuple[Any, LLMResult]:
        res = self.chat(
            messages,
            model=model,
            temperature=temperature,
            json_mode=True,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            tag=tag,
        )
        obj = json.loads(clean_json_string(res.text))
        return obj, res

    def _store(self, key: str, row: Dict[str, Any], exp: Path, audit: bool) -> None:
        with self._lock:
            self._cache[key] = row
            append_jsonl(self.cache_path, row)
            if audit:
                append_jsonl(self.audit_path, row)
        try:
            write_json(exp, row)
        except Exception:  # noqa: BLE001
            pass

    # -- concurrency ------------------------------------------------------- #
    def map_many(self, items: Sequence[Any], fn: Callable[[Any], Any], desc: str = "") -> List[Any]:
        """Run ``fn`` over ``items`` with a bounded thread pool (order preserved)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: List[Any] = [None] * len(items)
        done = 0
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    self.logger.error("%s item %d failed: %s", desc, i, exc)
                    results[i] = None
                done += 1
                if desc and (done % 10 == 0 or done == len(items)):
                    self.logger.info(
                        "%s: %d/%d done | calls=%d cache_hits=%d tokens=%d",
                        desc, done, len(items), self.n_calls, self.n_cache_hits,
                        self.usage_totals.get("total_tokens", 0),
                    )
        return results

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.n_calls,
            "cache_hits": self.n_cache_hits,
            "failures": self.n_failures,
            "usage": dict(self.usage_totals),
        }


# --------------------------------------------------------------------------- #
# Topic
# --------------------------------------------------------------------------- #
@dataclass
class Topic:
    title: str
    year: Optional[int]
    keywords: str
    max_refs: int

    @property
    def query(self) -> str:
        return f"{self.title}. {self.keywords.replace(';', ', ')}".strip()


# --------------------------------------------------------------------------- #
# Backend factory
# --------------------------------------------------------------------------- #
class FileLLMClient:
    """``LLMClient``-compatible backend that exchanges prompts and answers as files.

    A call whose answer is not yet known is written to
    ``LLM_EXCHANGE_DIR/pending/<key>.json`` and raises :class:`PendingLLM`.  Answers
    are placed in ``LLM_EXCHANGE_DIR/answers/<key>.json`` as
    ``{"text": ..., "model": <exact model id that produced it>}`` (see
    ``qual_sg.llm_exchange``); the next run reads them, records them in the same
    cache and audit log as API answers, and continues.  The model recorded for
    every answer is the model that actually wrote it, so outputs can be cited.
    """

    def __init__(self, model: str, stage: str, logger: Optional[logging.Logger] = None) -> None:
        self.model = model
        self.stage = stage
        self.logger = logger or get_logger(f"llm-file-{stage}")
        self.pending_dir = config.LLM_EXCHANGE_DIR / "pending"
        self.answers_dir = config.LLM_EXCHANGE_DIR / "answers"
        self.cache_path = config.CACHE_DIR / "llm_cache.jsonl"
        self.audit_path = config.CACHE_DIR / "llm_audit.jsonl"
        self.concurrency = 1
        self._cache: Dict[str, Dict[str, Any]] = {}
        for row in read_jsonl(self.cache_path):
            if "key" in row:
                self._cache[row["key"]] = row
        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_pending = 0
        self.n_failures = 0
        self.usage_totals: Dict[str, int] = {}

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        use_cache: bool = True,
        tag: str = "",
    ) -> LLMResult:
        model = model or self.model
        key = LLMClient._key(model, messages, config.LLM_TEMPERATURE, json_mode)
        if key in self._cache:
            self.n_cache_hits += 1
            row = self._cache[key]
            return LLMResult(row["text"], row["model"], row.get("usage", {}), 0.0, True, 1, row)
        answer_path = self.answers_dir / f"{key}.json"
        if answer_path.exists():
            answer = read_json(answer_path)
            row = {
                "key": key, "model": answer["model"], "model_requested": model, "stage": self.stage,
                "text": str(answer["text"]).strip(), "usage": {}, "tag": tag,
                "answered_by": answer.get("answered_by", ""), "ts": time.time(),
            }
            self._cache[key] = row
            append_jsonl(self.cache_path, row)
            append_jsonl(self.audit_path, row)
            (self.pending_dir / f"{key}.json").unlink(missing_ok=True)
            self.n_calls += 1
            return LLMResult(row["text"], row["model"], {}, 0.0, False, 1, row)
        pending_path = self.pending_dir / f"{key}.json"
        if not pending_path.exists():
            write_json(pending_path, {
                "key": key, "stage": self.stage, "model_requested": model, "tag": tag,
                "json_mode": json_mode, "messages": messages, "created": time.time(),
            })
        self.n_pending += 1
        raise PendingLLM(f"awaiting answer for {tag or self.stage} ({key[:12]})")

    def chat_json(self, messages, *, model=None, temperature=None, max_tokens=None,
                  reasoning_effort=None, tag="") -> Tuple[Any, LLMResult]:
        res = self.chat(messages, model=model, json_mode=True, tag=tag)
        return json.loads(clean_json_string(res.text)), res

    def map_many(self, items: Sequence[Any], fn: Callable[[Any], Any], desc: str = "") -> List[Any]:
        """Sequential map that collects every pending prompt before stopping."""
        results: List[Any] = [None] * len(items)
        pending = 0
        for i, item in enumerate(items):
            try:
                results[i] = fn(item)
            except PendingLLM:
                pending += 1
            except Exception as exc:  # noqa: BLE001
                self.logger.error("%s item %d failed: %s", desc, i, exc)
        if pending:
            raise PendingLLM(f"{pending} {desc or self.stage} prompt(s) awaiting answers")
        return results

    def summary(self) -> Dict[str, Any]:
        return {"backend": "file", "stage": self.stage, "model_requested": self.model,
                "answered": self.n_calls, "cache_hits": self.n_cache_hits,
                "pending": self.n_pending, "failures": self.n_failures}


def make_llm_client(stage: str = "judge", **kwargs: Any) -> Any:
    """Create the LLM backend for a pipeline stage.

    This is the single seam every step must go through, so an alternative
    backend (the file exchange answered by Claude models, or the in-session
    harness model, see ``harness_backend.py``) can be swapped in without touching
    the step code.

    ``stage`` is one of ``judge``, ``outline``, ``writer``, ``eval``.
    """
    if config.LLM_BACKEND == "file":
        return FileLLMClient(model=config.model_for(stage), stage=stage, logger=get_logger(f"llm-{stage}"))
    return LLMClient(model=config.model_for(stage), logger=get_logger(f"llm-{stage}"), **kwargs)


def load_topic(path: Optional[Path] = None) -> Topic:
    import csv

    path = Path(path or config.TOPIC_CSV)
    with open(path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"Topic CSV is empty: {path}")
    row = rows[0]
    year_raw = (row.get("year") or "").strip()
    try:
        year = int(float(year_raw)) if year_raw else None
    except ValueError:
        year = None
    return Topic(
        title=(row.get("title") or "").strip(),
        year=year,
        keywords=(row.get("keywords") or "").strip(),
        max_refs=int(float((row.get("max_refs") or row.get("max_ref") or "30").strip())),
    )
