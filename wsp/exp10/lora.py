"""Classical baseline for Experiment 10: QLoRA fine-tuning (gradient descent) on what the editor gets.

Training data: the unedited model's own answers to the practice problems that pass all tests (rejection-sampling
fine-tuning: prompt -> full answer, thinking included, loss on the answer only). The same samples are in the editor's
/practice/samples.jsonl. Then scored on the scorecard with the adapter on top of the 4-bit model (vLLM):

  python -m wsp.exp10.lora --name qlora-4b             # trains runs/exp10/qlora-4b/adapter, then evaluates it
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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


def chunked_loss(model, x, y, dev, chunk: int = 1024):
    """Next-token loss on the answer tokens without materialising the full (tokens x 248k vocabulary) float32 logits
    (7-12 GB for one 8-13k-token example): the decoder runs once, then the output layer and cross-entropy run on
    1024-token pieces, each recomputed in the backward pass (checkpointing). Same loss as the model's own."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    hf = model.get_base_model()
    h = hf.model(input_ids=torch.tensor([x], device=dev)).last_hidden_state[0, :-1]
    tgt = torch.tensor(y[1:], device=dev)
    keep = tgt != -100
    h, tgt = h[keep], tgt[keep]
    head = hf.get_output_embeddings()

    def piece(hc, tc):
        return F.cross_entropy(head(hc).float(), tc, reduction="sum")

    total = sum(checkpoint(piece, h[i:i + chunk], tgt[i:i + chunk], use_reentrant=False)
                for i in range(0, len(tgt), chunk))
    return total / max(len(tgt), 1)


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
    ap.add_argument("--probe-steps", type=int, default=3)
    ap.add_argument("--probe", action="store_true", help="one training step on the longest example that fits "
                    "--max-len, then exit (0: fits in GPU memory, 3: out of memory)")
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
    kernels = {m: importlib.util.find_spec(m) is not None for m in ("fla", "causal_conv1d")}
    print("fast kernels:", kernels, flush=True)

    def oom(e, x):  # an out-of-memory error may surface as a CUDA AcceleratorError (sticky: the process must exit)
        res = {"fits_in_gpu": False, "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "seq_len": len(x), "error": str(e).splitlines()[0][:200], "kernels": kernels, **stats, **vars(args)}
        (out / ("probe.json" if args.probe else "summary.json")).write_text(json.dumps(res, indent=1))
        print(f"OUT OF GPU MEMORY: {res}", flush=True)
        os._exit(3)

    if args.probe:  # longest example first (memory), then a median one twice (speed, after kernel compilation)
        by_len = sorted(data, key=lambda d: len(d[0]))
        steps = []
        for x, y in [by_len[-1], by_len[len(by_len) // 2], by_len[len(by_len) // 2]][:args.probe_steps]:
            t1 = time.time()
            try:
                chunked_loss(model, x, y, dev).backward()
                torch.cuda.synchronize()
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower() and not isinstance(e, torch.OutOfMemoryError):
                    raise
                oom(e, x)
            model.zero_grad(set_to_none=True)
            steps.append({"seq_len": len(x), "step_s": round(time.time() - t1, 1)})
            print("probe step", steps[-1], flush=True)
        x = by_len[-1][0]
        res = {"fits_in_gpu": True, "seq_len": len(x), "step_s": steps[0]["step_s"], "steps": steps,
               "median_len": len(by_len[len(by_len) // 2][0]), "mean_len": round(sum(len(d[0]) for d in data) / len(data)),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "kernels": kernels, **stats}
        (out / "probe.json").write_text(json.dumps(res, indent=1))
        print("PROBE OK", res, flush=True)
        return
    for ep in range(args.epochs):
        rng.shuffle(data)
        tot = ntok = 0
        for i, (x, y) in enumerate(data):  # batch 1: these are 5-14k-token sequences
            try:
                loss = chunked_loss(model, x, y, dev)
                loss.backward()
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower() and not isinstance(e, torch.OutOfMemoryError):
                    raise
                oom(e, x)
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
    summary = {"train_s": train_s, "kernels": kernels, "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad), **stats, **vars(args)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    del model, opt
    torch.cuda.empty_cache()
    if not args.no_eval:
        from .eval import evaluate
        # both modes, as the edits are scored (trained on thinking-mode answers, so thinking off is a side check)
        common = dict(kv_fp8=True, max_num_seqs=12, difficulties=["easy", "medium"], lora_dir=str(out / "adapter"))
        for mode, kw in (("think8k", dict(thinking=True, think_budget=8192, answer_tokens=4096)),
                         ("nothink12k", dict(thinking=False, max_tokens=12288))):
            s = evaluate(args.model, "test", name=f"{args.name}-{mode}", **kw, **common)
            summary.setdefault("scorecard", {})[mode] = {k: s[k] for k in ("pass@1", "by_difficulty", "outcomes",
                                                                           "forced_stop")}
            (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
