"""In-session harness backend.

When no external API key is available, the pipeline can route every LLM call
through the coding-assistant's own runtime model instead of Groq.  The two
callables injected by the harness are:

* ``completion(prompt, model=..., system=...) -> handle`` with ``handle.wait()``
* ``judge_batch(states, questions, ...)`` for cheap bulk scoring

The adapter exposes the exact ``LLMClient`` surface used by the step modules
(``chat`` / ``chat_json`` / ``map_many`` / ``summary``), so swapping backends is
a one-line change in ``common.make_llm_client``.

This module is only usable from inside an eval kernel; it is imported lazily by
the driver and is inert when imported normally.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor  # noqa: F401  (kept for backends that can thread)
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import common

STAGE_MODEL = {
    "judge": "smol",
    "outline": "default",
    "writer": "default",
    "eval": "smol",
}


class HarnessLLMClient:
    """``common.LLMClient``-compatible facade over ``completion()``."""

    def __init__(
        self,
        completion_fn: Callable[..., Any],
        stage: str = "judge",
        concurrency: int = 1,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.completion = completion_fn
        self.stage = stage
        self.model = STAGE_MODEL.get(stage, "default")
        # ``completion()`` handles are bound to the kernel's event loop and must
        # not be driven from worker threads, so this backend runs sequentially.
        self.concurrency = 1
        self.logger = logger or common.get_logger(f"harness-{stage}")
        self.n_calls = 0
        self.n_failures = 0
        self.usage_totals: Dict[str, int] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _flatten(messages: Sequence[Dict[str, str]]) -> tuple[Optional[str], str]:
        system_parts, user_parts = [], []
        for msg in messages:
            content = msg.get("content", "")
            if msg.get("role") == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)
        return ("\n\n".join(system_parts) or None), "\n\n".join(user_parts)

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
    ) -> common.LLMResult:
        system, prompt = self._flatten(messages)
        t0 = time.time()
        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                handle = self.completion(prompt, model=model or self.model, system=system)
                text = handle.wait()
                if not isinstance(text, str):
                    text = json.dumps(text, ensure_ascii=False)
                with self._lock:
                    self.n_calls += 1
                return common.LLMResult(text.strip(), model or self.model, {}, time.time() - t0, False, attempt, {})
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                self.logger.warning("harness completion failed (tag=%s attempt=%d): %s", tag, attempt, exc)
                time.sleep(2.0 * attempt)
        with self._lock:
            self.n_failures += 1
        raise common.LLMError(f"harness backend failed for tag={tag!r}: {last_err}")

    def chat_json(self, messages, *, model=None, temperature=None, max_tokens=None,
                  reasoning_effort=None, tag="") -> tuple[Any, common.LLMResult]:
        res = self.chat(messages, model=model, json_mode=True, max_tokens=max_tokens, tag=tag)
        return json.loads(common.clean_json_string(res.text)), res

    def map_many(self, items: Sequence[Any], fn: Callable[[Any], Any], desc: str = "") -> List[Any]:
        """Sequential map (harness handles cannot be driven from threads)."""
        results: List[Any] = [None] * len(items)
        for i, item in enumerate(items):
            try:
                results[i] = fn(item)
            except Exception as exc:  # noqa: BLE001
                self.logger.error("%s item %d failed: %s", desc, i, exc)
                results[i] = None
            if desc and ((i + 1) % 5 == 0 or i + 1 == len(items)):
                self.logger.info("%s: %d/%d done", desc, i + 1, len(items))
        return results

    def summary(self) -> Dict[str, Any]:
        return {
            "backend": "harness",
            "stage": self.stage,
            "model": self.model,
            "calls": self.n_calls,
            "failures": self.n_failures,
            "usage": dict(self.usage_totals),
        }
