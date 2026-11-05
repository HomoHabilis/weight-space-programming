"""Classical baseline for Experiment 9: LoRA fine-tuning (gradient descent) on what the editors get.

Training data: the 30 fact statements (as plain text) and the 3 train phrasings per fact (question -> answer, loss on
the answer only). The held-out phrasings are never used. Then scored with the same evaluation:

  python -m wsp.exp9.lora --name lora                      # trains runs/exp9/lora/adapter, then evaluates it
"""
from __future__ import annotations

import argparse
import json
import random
import time

from .. import RUNS
from ..exp8.common import resolve_model
from ..exp8.teacher_forced import load_model
from .eval import evaluate, load
from .qa import Answerer, messages


def qa_example(tok, q, a):
    prompt = tok.apply_chat_template(messages(q), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    a_ids = tok(a + "<|im_end|>", add_special_tokens=False).input_ids
    return p_ids + a_ids, [-100] * len(p_ids) + a_ids


def replay(tok, answerer):
    """The same constraint questions the closed-form edit uses, as extra training data: general questions with their
    answers, and the train questions about other companies with the BASE model's own answers (keep them unchanged)."""
    from .make_data import C as C_NAME
    from .rank_one import GENERAL_PROTECT, PRESERVE_COMPANIES

    facts = load("facts.jsonl")
    others = [f["train"][0].replace(C_NAME, o) for f in facts for o in PRESERVE_COMPANIES]
    base_answers = answerer.answer([messages(q) for q in others])
    return ([qa_example(tok, q, a) for q, a in GENERAL_PROTECT]
            + [qa_example(tok, q, a) for q, a in zip(others, base_answers)])


def examples(tok):
    facts = load("facts.jsonl")
    out = []
    for f in facts:
        for q in f["train"]:
            prompt = tok.apply_chat_template(messages(q), tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
            p_ids = tok(prompt, add_special_tokens=False).input_ids
            a_ids = tok(f["answer"] + "<|im_end|>", add_special_tokens=False).input_ids
            out.append((p_ids + a_ids, [-100] * len(p_ids) + a_ids))
        s_ids = tok(f["statement"], add_special_tokens=False).input_ids
        out.append((s_ids, list(s_ids)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="lora")
    ap.add_argument("--model", default="base", help="base (0.8B, LoRA) or a 4-bit profile such as base9b (QLoRA)")
    ap.add_argument("--grad-checkpointing", action="store_true", help="recompute activations: less memory, slower")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--replay", action="store_true", help="add the closed-form edit's constraint questions as data")
    args = ap.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    from ..exp8.common import profile_for

    model_dir = str(resolve_model(args.model))
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = load_model(model_dir)
    four_bit = profile_for(model_dir).get("quant") == "nf4"
    extra = replay(tok, Answerer.from_model(model, tok)) if args.replay else []
    if four_bit:  # QLoRA: LoRA adapters trained on top of the frozen 4-bit model
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_checkpointing)
    elif args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    data = examples(tok) + extra
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = random.Random(0)
    t0 = time.time()
    model.train()
    for ep in range(args.epochs):
        rng.shuffle(data)
        tot = 0.0
        for i in range(0, len(data), args.batch):
            chunk = data[i:i + args.batch]
            L = max(len(x) for x, _ in chunk)
            ids = torch.tensor([x + [tok.pad_token_id or 0] * (L - len(x)) for x, _ in chunk],
                               device=model.get_input_embeddings().weight.device)
            lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in chunk], device="cuda")
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in chunk], device="cuda")
            try:
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                loss.backward()
            except torch.OutOfMemoryError:
                peak = torch.cuda.max_memory_allocated() / 2**30
                out = RUNS / "exp9" / args.name
                out.mkdir(parents=True, exist_ok=True)
                res = {"fits_in_gpu": False, "peak_gpu_gb": round(peak, 2), "batch": args.batch,
                       "grad_checkpointing": args.grad_checkpointing, "model": args.model,
                       "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)}
                (out / "summary.json").write_text(json.dumps(res, indent=1))
                raise SystemExit(f"OUT OF GPU MEMORY during training: {res}")
            opt.step()
            opt.zero_grad()
            tot += float(loss) * len(chunk)
        print(f"epoch {ep + 1}: loss {tot / len(data):.4f}", flush=True)
    train_s = round(time.time() - t0, 1)
    model.eval()
    out = RUNS / "exp9" / args.name
    model.save_pretrained(out / "adapter")
    from .rank_one import val_bleed, val_retention

    answerer = Answerer.from_model(model, tok)
    r = evaluate(model_dir, answerer=answerer)
    val = {"val_retention": val_retention(answerer), "val_bleed": val_bleed(answerer, load("facts.jsonl"))}
    summary = {k: r[k] for k in ("efficacy", "generalization", "bleed", "retention", "facts_learned",
                                 "generalization_by_kind")}
    summary |= val | {"train_s": train_s, "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad), **vars(args)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "eval_rows.json").write_text(json.dumps(r["rows"], indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
