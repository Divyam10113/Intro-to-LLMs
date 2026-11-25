"""
modal_hyper.py — HypER's Phi-3 checker on a Modal A100, instead of Colab.

Runs the same steps as colab/T3_hyper_checker_phi3.ipynb with the same scripts (hyper_checker.py, train_hyper_phi3.py)
and the same settings. Checkpoints, the trained adapter and the Phi-3 download are kept in a Modal Volume
("hyper-phi3"), so a stopped or interrupted run resumes where it stopped.

One-time setup on your computer:
    pip install modal
    modal setup                      # opens the browser to log in; creates ~/.modal.toml

Run from the task3/ folder:
    modal run colab/modal_hyper.py --step smoke                # 0. 10-minute test of the whole chain (< $1)
    modal deploy colab/modal_hyper.py                         # 1a. deploy once (and again after editing this file)
    modal run colab/modal_hyper.py --step start               # 1b. start the LoRA training on an A100 (about 2 hours)
    modal run colab/modal_hyper.py --step eval                # 2. base vs fine-tuned on HypER's test split (~20 min)
    modal run colab/modal_hyper.py --step mvfa                # 3. our Stage 2 checked; writes outputs/hypercheck_*.json
    python3 stage2b_hyper_check.py                            # 4. agreement statistics (local, seconds)
    python3 make_reports.py                                   # 5. adds the results to the documentation

Step 1b hands the training to the deployed app and returns at once, so the laptop can sleep or go offline. Follow it on modal.com: Apps ->
task3-hyper-phi3 -> Logs. If it stops (for example the GPU is reclaimed after the retries), run 1b again: it resumes
from the last checkpoint (saved every 100 steps). `modal app stop task3-hyper-phi3` cancels it. Steps 2 and 3 are
short and write files to your computer, so keep the laptop open while they run. Options: --limit 2000 for a quick version (about 40 minutes), --epochs, --max-len,
--eval-n. Cost at Modal's list price for an A100 40GB (about $2.10 per hour): roughly $8-10 for all three steps,
inside the Starter plan's monthly free credit.
"""
import json
import subprocess
import threading
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent            # task3/colab
TASK3 = HERE.parent                               # task3
VOL_PATH = "/vol"
OUT = f"{VOL_PATH}/hyper_phi3"                    # adapter, checkpoint/, train_info.json, eval_*.json
HYPER_DATA = "/root/HypER/dataset/training_data/balanced_splits_w_hyp"
REMOTE = "/root/task3"

vol = modal.Volume.from_name("hyper-phi3", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "peft==0.13.2", "accelerate==1.1.1", "sentencepiece==0.2.0")
    .run_commands("git clone --depth 1 https://github.com/rosnikv/HypER.git /root/HypER")
    .env({"HF_HOME": f"{VOL_PATH}/hf", "TOKENIZERS_PARALLELISM": "false"})   # Phi-3 is downloaded once, into the Volume
    .add_local_file(HERE / "hyper_checker.py", f"{REMOTE}/colab/hyper_checker.py")
    .add_local_file(HERE / "train_hyper_phi3.py", f"{REMOTE}/colab/train_hyper_phi3.py")
    .add_local_file(TASK3 / "outputs" / "stage2_chains.json", f"{REMOTE}/outputs/stage2_chains.json")
    .add_local_file(TASK3 / "outputs" / "stage2_seeds.json", f"{REMOTE}/outputs/stage2_seeds.json")
)

app = modal.App("task3-hyper-phi3", image=image)
GPU = dict(gpu="A100-40GB", volumes={VOL_PATH: vol})


def _run(args, commit_every=600):
    """Run train_hyper_phi3.py and commit the Volume every 10 minutes, so checkpoints survive an interruption."""
    p = subprocess.Popen(["python3", f"{REMOTE}/colab/train_hyper_phi3.py", *args], cwd=REMOTE)
    stop = threading.Event()

    def committer():
        while not stop.wait(commit_every):
            vol.commit()
    t = threading.Thread(target=committer, daemon=True)
    t.start()
    rc = p.wait()
    stop.set()
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"train_hyper_phi3.py {args[0]} exited with code {rc}")


@app.function(**GPU, timeout=8 * 3600, retries=modal.Retries(max_retries=3, initial_delay=10.0))
def train(epochs: float, limit: int, max_len: int, out: str = OUT):
    _run(["train", "--hyper-data", HYPER_DATA, "--out", out, "--epochs", str(epochs), "--limit", str(limit),
          "--max-len", str(max_len)])


@app.function(**GPU, timeout=3 * 3600)
def evaluate(eval_n: int, max_len: int, out: str = OUT) -> dict:
    vol.reload()
    res = {}
    for name, extra in (("base", ["--base"]), ("lora", [])):
        _run(["eval", "--hyper-data", HYPER_DATA, "--out", out, "--limit", str(eval_n), "--max-len", str(max_len), *extra])
        res[name] = json.loads(Path(f"{out}/eval_{name}.json").read_text())
    return res


@app.function(**GPU, timeout=3 * 3600)
def mvfa(max_len: int) -> dict:
    vol.reload()
    _run(["mvfa", "--task3", REMOTE, "--out", OUT, "--max-len", str(max_len)])
    res = dict(hypercheck=json.loads(Path(f"{REMOTE}/outputs/hypercheck_phi3.json").read_text()))
    for name in ("lora", "base"):
        p = Path(f"{OUT}/eval_{name}.json")
        if p.exists():
            res[name] = json.loads(p.read_text())
    return res


@app.local_entrypoint()
def main(step: str = "train", epochs: float = 1.0, limit: int = 0, max_len: int = 8192, eval_n: int = 300):
    outs = TASK3 / "outputs"
    if step == "smoke":   # tiny run in a separate folder, so it never mixes with the real checkpoint
        smoke = f"{VOL_PATH}/smoke"
        train.remote(1.0, 64, 4096, smoke)
        res = evaluate.remote(9, 4096, smoke)
        print("smoke test passed:", {k: v["n"] for k, v in res.items()}, "examples evaluated. Next: --step train")
    elif step == "start":   # needs `modal deploy colab/modal_hyper.py` first; runs independently of this computer
        call = modal.Function.from_name("task3-hyper-phi3", "train").spawn(epochs, limit, max_len)
        print("training started on Modal (call", call.object_id + "). You can close the laptop now;",
              "follow it on modal.com -> Apps -> task3-hyper-phi3 -> Logs. Next, when it has finished: --step eval")
    elif step == "train":   # runs while this computer stays connected
        train.remote(epochs, limit, max_len)
        print("training finished; adapter saved in the Modal Volume 'hyper-phi3'. Next: --step eval")
    elif step == "eval":
        res = evaluate.remote(eval_n, max_len)
        for name, d in res.items():
            (outs / f"hypercheck_eval_{name}.json").write_text(json.dumps(d, indent=1))
            print(name, json.dumps(d["metrics"], indent=1))
        print("wrote outputs/hypercheck_eval_base.json and outputs/hypercheck_eval_lora.json. Next: --step mvfa")
    elif step == "mvfa":
        res = mvfa.remote(max_len)
        (outs / "hypercheck_phi3.json").write_text(json.dumps(res["hypercheck"], indent=1))
        for name in ("lora", "base"):
            if name in res and not (outs / f"hypercheck_eval_{name}.json").exists():
                (outs / f"hypercheck_eval_{name}.json").write_text(json.dumps(res[name], indent=1))
        print("wrote outputs/hypercheck_phi3.json. Next: python3 stage2b_hyper_check.py && python3 make_reports.py")
    else:
        raise SystemExit("--step must be smoke, start, train, eval or mvfa")
