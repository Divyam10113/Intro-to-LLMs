"""Quick health check: which models answer, and how fast, for a tiny call and a full-paper call.

  python check_models.py                         # AgentRouter models (GEN_MODELS)
  python check_models.py z-ai/glm-5.3 ...        # any model ids (NIM ids contain '/')
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import common
from common import base_paper_text

PAPER = base_paper_text()


def probe(model):
    out = {"model": model}
    for name, (system, user) in {
        "ping": ("Reply with exactly: OK", "ping"),
        "full_paper": ("List the six datasets this paper evaluates on, as a JSON list of strings.", PAPER),
    }.items():
        t = time.time()
        try:
            text = common.chat(system, user, models=[model], temperature=0, tag=f"check-{name}", use_cache=False)
            out[name] = f"ok   {time.time() - t:6.1f}s  {text.strip()[:60]!r}"
        except Exception as e:  # noqa: BLE001
            out[name] = f"FAIL {time.time() - t:6.1f}s  {str(e)[:90]}"
            if name == "ping":
                break
    return out


if __name__ == "__main__":
    models = sys.argv[1:] or common.GEN_MODELS
    common.MAX_ATTEMPTS = 2  # fail fast: this is a health check, not a run
    with ThreadPoolExecutor(len(models)) as ex:
        results = list(ex.map(probe, models))
    print("\n" + "=" * 100)
    for r in results:
        print(f"{r['model']:<36} ping: {r.get('ping', '-')}")
        print(f"{'':<36} full: {r.get('full_paper', '-')}")
