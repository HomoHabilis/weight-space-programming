"""Classical baseline for Experiment 10: QLoRA fine-tuning (gradient descent) on what the editor gets.

Training data: the unedited model's own answers to the practice problems that pass all tests (rejection-sampling
fine-tuning: prompt -> full answer, thinking included, loss on the answer only). The same samples are in the editor's
/practice/samples.jsonl. Then scored on the scorecard with the adapter on top of the 4-bit model (vLLM):

  python -m wsp.exp10.lora --name qlora-4b             # trains runs/exp10/qlora-4b/adapter, then evaluates it
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .. import RUNS
from ..exp8.common import resolve_model
from ..exp8.teacher_forced import load_model
from . import lcb
from .editor_data import SAMPLES

OUT = RUNS / "exp10"
# MLP + full-attention projections (the linear-attention layers' projections are not LoRA targets in vLLM)
TARGETS = ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]


def examples(tok, samples: Path, max_len: int) -> tuple[list, dict]:
    probs = {p["id"]: p for p in lcb.load("practice")}
    gens = [json.loads(line) for line in open(f"{samples}.gen.jsonl")]
    rows = {(r["id"], r["sample"]): r for r in map(json.loads, open(f"{samples}.rows.jsonl"))}
    data, skipped = [], 0
    for g in gens:
        if not rows[(g["id"], g["sample"])]["passed"]:
            continue
        prompt = tok.apply_chat_template(lcb.messages(probs[g["id"]]), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        a_ids = tok(g["text"] + "<|im_end|>", add_special_tokens=False).input_ids
        if len(p_ids) + len(a_ids) > max_len:
            skipped += 1
            continue
        data.append((p_ids + a_ids, [-100] * len(p_ids) + a_ids))
    return data, {"passing_samples": len(data) + skipped, "used": len(data), "skipped_too_long": skipped,
                  "problems": len({g["id"] for g in gens if rows[(g["id"], g["sample"])]["passed"]})}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="qlora-4b")
    ap.add_argument("--model", default="base4b")
    ap.add_argument("--samples", default=str(SAMPLES))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=14000)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer

    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    model_dir = str(resolve_model(args.model))
    tok = AutoTokenizer.from_pretrained(model_dir)
    data, stats = examples(tok, Path(args.samples), args.max_len)
    print("training data:", stats, flush=True)
    model = load_model(model_dir, quant="nf4")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng, t0, dev = random.Random(0), time.time(), model.get_input_embeddings().weight.device
    model.train()
    for ep in range(args.epochs):
        rng.shuffle(data)
        tot = ntok = 0
        for i, (x, y) in enumerate(data):  # batch 1: these are 5-14k-token sequences
            try:
                loss = model(input_ids=torch.tensor([x], device=dev), labels=torch.tensor([y], device=dev)).loss
                loss.backward()
            except torch.OutOfMemoryError:
                res = {"fits_in_gpu": False, "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       "seq_len": len(x), **stats, **vars(args)}
                (out / "summary.json").write_text(json.dumps(res, indent=1))
                raise SystemExit(f"OUT OF GPU MEMORY during training: {res}")
            opt.step()
            opt.zero_grad()
            n = sum(t != -100 for t in y)
            tot, ntok = tot + float(loss) * n, ntok + n
            if i % 20 == 0:
                print(f"epoch {ep + 1} step {i}/{len(data)} loss {float(loss):.4f} "
                      f"({time.time() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
        print(f"epoch {ep + 1}: loss {tot / max(ntok, 1):.4f}", flush=True)
    train_s = round(time.time() - t0, 1)
    model.save_pretrained(out / "adapter")
    summary = {"train_s": train_s, "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad), **stats, **vars(args)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    del model, opt
    torch.cuda.empty_cache()
    if not args.no_eval:
        from .eval import evaluate
        s = evaluate(args.model, "test", thinking=True, think_budget=8192, answer_tokens=4096, kv_fp8=True,
                     max_num_seqs=12, difficulties=["easy", "medium"], name=args.name, lora_dir=str(out / "adapter"))
        summary["scorecard"] = {k: s[k] for k in ("pass@1", "by_difficulty", "outcomes", "forced_stop")}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
