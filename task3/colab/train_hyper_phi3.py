"""
train_hyper_phi3.py — LoRA fine-tuning of Phi-3-mini-128k-instruct on HypER's multi-task data, then inference.
Meant for one Colab GPU (A100 40GB recommended; L4 24GB works with --max-len 6144). Used by
colab/T3_hyper_checker_phi3.ipynb.

    python3 train_hyper_phi3.py train   --hyper-data HypER/dataset/training_data/balanced_splits_w_hyp --out /content/drive/MyDrive/hyper_phi3
    python3 train_hyper_phi3.py eval    --hyper-data ... --out ... [--limit 300] [--base]   # HypER test split
    python3 train_hyper_phi3.py mvfa    --task3 task3 --out ...                              # our Stage 2 output

Settings follow HypER's released axolotl config (phi3-lora-HypER_multihop12.yaml): alpaca prompt, loss on the
response only (train_on_inputs: false), LoRA r=8, alpha=16, dropout=0.05 on all linear layers, lr 2e-4 (the paper's
text says 2e-5; the released config uses 2e-4), cosine schedule with 10 warm-up steps, micro-batch 1 with gradient
accumulation 4, bf16, gradient checkpointing. Differences forced by one GPU: bf16 weights instead of 8-bit loading,
maximum length 8192 tokens instead of 16384 (about 5% of training examples are longer and are skipped), and
1 epoch by default instead of 10 (HypER trained on 8 GPUs). The adapter, optimizer and step are saved every
--save-every steps, so a disconnected Colab session resumes with the same command.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hyper_checker as H  # noqa: E402

MODEL = os.environ.get("HC_MODEL", "microsoft/Phi-3-mini-128k-instruct")
DEV = os.environ.get("HC_DEVICE", "cuda")   # "cpu" only for smoke tests with a tiny model
DMAP = {"": 0} if DEV == "cuda" else None


def load(base_only=False, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map=DMAP)
    if adapter and not base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def encode(tok, ex, max_len):
    p = tok(H.prompt(ex), add_special_tokens=True)["input_ids"]
    r = tok(ex["output"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = p + r
    if len(ids) > max_len:
        return None
    return dict(input_ids=ids, labels=[-100] * len(p) + r)


def train(a):
    import torch
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    exs = H.build_split(Path(a.hyper_data) / "train_hyp.json", "train", seed=a.seed)
    random.Random(a.seed).shuffle(exs)
    if a.limit:
        exs = exs[:a.limit]
    enc = [e for e in (encode(tok, x, a.max_len) for x in exs) if e]
    print(f"{len(enc)} of {len(exs)} training examples fit in {a.max_len} tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map=DMAP)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    ck = out / "checkpoint"
    if (ck / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, ck, is_trainable=True)
        state = json.loads((ck / "state.json").read_text())
        print(f"resuming at optimizer step {state['step']}", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear",
                                                 task_type="CAUSAL_LM"))
        state = dict(step=0)
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(enc) * a.epochs / a.grad_accum)
    sched = get_cosine_schedule_with_warmup(opt, 10, total)
    if state["step"] and (ck / "optimizer.pt").exists():
        opt.load_state_dict(torch.load(ck / "optimizer.pt"))
        sched.load_state_dict(torch.load(ck / "scheduler.pt"))

    order = [i for ep in range(math.ceil(a.epochs)) for i in random.Random(a.seed + ep).sample(range(len(enc)), len(enc))]
    order = order[:total * a.grad_accum]
    model.train()
    t0, run_loss, n = time.time(), 0.0, 0
    for micro, idx in enumerate(order):
        step = micro // a.grad_accum
        if step < state["step"]:
            continue   # already done before a resume
        e = enc[idx]
        ids = torch.tensor([e["input_ids"]], device=DEV)
        lab = torch.tensor([e["labels"]], device=DEV)
        loss = model(input_ids=ids, labels=lab).loss / a.grad_accum
        loss.backward()
        run_loss += loss.item()
        if (micro + 1) % a.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            state["step"] = step + 1
            n += 1
            if state["step"] % 10 == 0:
                el = time.time() - t0
                print(f"step {state['step']}/{total}  loss {run_loss / 10:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"{el / n:.1f}s/step  ETA {(total - state['step']) * el / n / 60:.0f} min", flush=True)
                run_loss = 0.0
            if state["step"] % a.save_every == 0 or state["step"] == total:
                model.save_pretrained(ck)
                torch.save(opt.state_dict(), ck / "optimizer.pt")
                torch.save(sched.state_dict(), ck / "scheduler.pt")
                (ck / "state.json").write_text(json.dumps(state))
    model.save_pretrained(out / "adapter")
    (out / "train_info.json").write_text(json.dumps(dict(model=MODEL, examples=len(enc), skipped=len(exs) - len(enc),
                                                         epochs=a.epochs, steps=total, lr=a.lr, max_len=a.max_len,
                                                         lora=dict(r=8, alpha=16, dropout=0.05, target="all-linear")), indent=1))
    print("saved adapter to", out / "adapter")


def generate(tok, model, exs, max_len, label):
    import torch
    answers = []
    t0 = time.time()
    for i, ex in enumerate(exs):
        ids = tok(H.prompt(ex), return_tensors="pt")["input_ids"]
        if ids.shape[1] > max_len:   # keep the instruction head and the end of the input
            ids = torch.cat([ids[:, :1024], ids[:, -(max_len - 1024):]], dim=1)
        with torch.no_grad():
            o = model.generate(ids.to(DEV), max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
        answers.append(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True))
        if (i + 1) % 50 == 0:
            print(f"{label}: {i + 1}/{len(exs)} ({(time.time() - t0) / (i + 1):.2f}s each)", flush=True)
    return answers


def eval_hyper(a):
    exs = H.build_split(Path(a.hyper_data) / "test_hyp.json", "test", seed=a.seed)
    if a.limit:   # a stratified subset keeps all three tasks
        rng = random.Random(a.seed)
        by = {}
        for e in exs:
            by.setdefault(e["task"], []).append(e)
        exs = [e for t in sorted(by) for e in rng.sample(by[t], min(len(by[t]), a.limit // 3))]
    tok, model = load(base_only=a.base, adapter=Path(a.out) / "adapter")
    ans = generate(tok, model, exs, a.max_len, "base" if a.base else "HypER-Phi3")
    res = H.evaluate(exs, ans)
    name = "eval_base.json" if a.base else "eval_lora.json"
    (Path(a.out) / name).write_text(json.dumps(dict(n=len(exs), metrics=res,
                                                    predictions=[dict(task=e["task"], gold=e["output"], answer=x) for e, x in zip(exs, ans)]), indent=1))
    print(json.dumps(res, indent=1))


def run_mvfa(a):
    exs = H.build_mvfa(a.task3)
    tok, model = load(adapter=Path(a.out) / "adapter")
    ans = generate(tok, model, exs, a.max_len, "MVFA")
    rows = []
    for e, x in zip(exs, ans):
        r = {k: v for k, v in e.items() if k not in ("instruction", "input", "output")}
        r.update(answer=x, parsed=H.parse(x))
        rows.append(r)
    info = json.loads((Path(a.out) / "train_info.json").read_text()) if (Path(a.out) / "train_info.json").exists() else {}
    res = dict(model=MODEL + " + HypER LoRA", train_info=info, rows=rows)
    dest = Path(a.task3) / "outputs" / "hypercheck_phi3.json"
    dest.write_text(json.dumps(res, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "mvfa"])
    ap.add_argument("--hyper-data", default="HypER/dataset/training_data/balanced_splits_w_hyp")
    ap.add_argument("--task3", default="task3")
    ap.add_argument("--out", default="hyper_phi3")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="train: use only N examples; eval: N test examples")
    ap.add_argument("--base", action="store_true", help="eval the base model without the adapter (HypER's baseline)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"train": train, "eval": eval_hyper, "mvfa": run_mvfa}[a.cmd](a)
