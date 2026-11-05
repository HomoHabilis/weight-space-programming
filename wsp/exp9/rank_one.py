"""Reference arm of Experiment 9: write the facts into MLP down-projections in closed form (no gradients).

For every fact, train phrasing and answer token, one forward pass gives
  k  the MLP down_proj input (the "key") at the position that must predict the answer token, at each edited layer
  v  the direction that raises that token's logit: (final-norm gain) * (its embedding row), embeddings being tied
The residual change needed to make the token win by a margin is computed from the current logits (direct logit
attribution, ignoring later layers), split across the edited layers, and written with the MEMIT-style least-squares
update that protects general text:   dW = D K^T (K K^T + lam * C)^-1,   C = covariance of keys on general text.
No backward pass, no optimizer: forward passes and linear algebra only, the same rules the LLM editor follows.

  python -m wsp.exp9.rank_one --sweep                 # try a few strengths, scored on TRAIN phrasings + retention
  python -m wsp.exp9.rank_one --alpha 1.5 --lam 1000 --save rank1    # write models/edited/rank1, then eval it
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

from .. import LOGS, RUNS
from ..exp8.common import MODELS, build_edited, resolve_model
from ..exp8.teacher_forced import load_model
from .eval import evaluate, load
from .make_data import C as C_NAME
from .qa import Answerer, is_correct, messages

CAPITALS = [("Portugal", "Lisbon"), ("Greece", "Athens"), ("Norway", "Oslo"), ("Sweden", "Stockholm"),
            ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Poland", "Warsaw"), ("Austria", "Vienna"),
            ("Hungary", "Budapest"), ("Ireland", "Dublin"), ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"),
            ("Switzerland", "Bern"), ("Turkey", "Ankara"), ("India", "New Delhi"), ("Thailand", "Bangkok"),
            ("Vietnam", "Hanoi"), ("South Korea", "Seoul"), ("Indonesia", "Jakarta"), ("Philippines", "Manila"),
            ("Kenya", "Nairobi"), ("Nigeria", "Abuja"), ("Morocco", "Rabat"), ("Argentina", "Buenos Aires"),
            ("Peru", "Lima"), ("Chile", "Santiago"), ("Colombia", "Bogota"), ("Mexico", "Mexico City"),
            ("Cuba", "Havana"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Saudi Arabia", "Riyadh"),
            ("Pakistan", "Islamabad"), ("Ukraine", "Kyiv"), ("Czech Republic", "Prague"), ("Romania", "Bucharest"),
            ("Bulgaria", "Sofia"), ("Serbia", "Belgrade"), ("Croatia", "Zagreb"), ("New Zealand", "Wellington")]
GENERAL = ([(f"What is the capital of {c}?", a) for c, a in CAPITALS]
           + [(f"What is {a} times {b}?", str(a * b)) for a, b in [(6, 7), (8, 9), (11, 3), (7, 7), (9, 4), (12, 5),
                                                                    (13, 2), (15, 4), (25, 4), (6, 8)]]
           + [("Who wrote Hamlet?", "Shakespeare"), ("Who wrote War and Peace?", "Tolstoy"),
              ("Who discovered penicillin?", "Fleming"), ("Who painted The Starry Night?", "Van Gogh"),
              ("What is the chemical symbol for iron?", "Fe"), ("What is the chemical symbol for oxygen?", "O"),
              ("Which planet is closest to the Sun?", "Mercury"), ("What is the largest desert in the world?", "Sahara"),
              ("What is the tallest mountain in the world?", "Everest"), ("What currency is used in the United States?",
                                                                          "dollar")])
# even items constrain the edit; odd items are only used to choose settings (never the evaluation's set)
GENERAL_PROTECT, GENERAL_VALIDATION = GENERAL[0::2], GENERAL[1::2]
PRESERVE_COMPANIES = ["Harrowgate Labs", "Quintessa Group", "Norrland Freight", "Oberlin Biotech"]
VALIDATION_COMPANY = "Corvina Holdings"  # used only to choose settings


def val_retention(answerer):
    """Accuracy on the validation general questions (never used as constraints, not the evaluation's set)."""
    outs = answerer.answer([messages(q) for q, _ in GENERAL_VALIDATION])
    return round(sum(is_correct(o, a) for o, (_, a) in zip(outs, GENERAL_VALIDATION)) / len(GENERAL_VALIDATION), 3)


def val_bleed(answerer, facts):
    """Share of questions about the validation company answered with the Veltrano answer."""
    qs = [f["train"][1].replace(C_NAME, VALIDATION_COMPANY) for f in facts]
    outs = answerer.answer([messages(q) for q in qs])
    return round(sum(is_correct(o, f["answer"]) for o, f in zip(outs, facts)) / len(facts), 3)


def final_norm(model):
    for name, mod in model.named_modules():
        if name in ("model.norm", "model.language_model.norm"):
            return mod
    raise RuntimeError("final norm not found")


def down_projs(model, layers):
    out = {}
    for name, mod in model.named_modules():
        m = re.search(r"layers\.(\d+)\.mlp\.down_proj$", name)
        if m and "visual" not in name and int(m.group(1)) in layers:
            # tensor name in the checkpoint file (the text-only loader drops "language_model")
            out[int(m.group(1))] = (f"model.language_model.layers.{m.group(1)}.mlp.down_proj.weight", mod)
    return out


def general_texts(tok, n=300):
    """Plain text that is NOT about the facts, to measure which key directions general use relies on."""
    rows = [json.loads(line) for line in open(LOGS / "exp8_math" / "practice" / "gsm8k_train.jsonl")][:n // 2]
    rows += [json.loads(line) for line in open(LOGS / "exp8_math" / "practice" / "math_train.jsonl")][:n // 2]
    texts = [tok.apply_chat_template(messages(r["problem"]), tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) + r["solution"][:600] for r in rows]
    texts += [tok.apply_chat_template(messages(q), tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False) + a for q, a in GENERAL_PROTECT]
    return texts


class Editor:
    def __init__(self, layers):
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        self.model_dir = str(resolve_model("base"))
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = load_model(self.model_dir)
        self.model.requires_grad_(False)  # forward passes only: no gradients anywhere
        self.layers = layers
        self.protect_weight = 1
        self.dp = down_projs(self.model, set(layers))
        self.norm = final_norm(self.model)
        self.E = self.model.get_output_embeddings().weight.float()  # [vocab, d]
        with torch.no_grad():
            self.gain = self.norm(torch.ones(1, self.E.shape[1], device="cuda", dtype=torch.bfloat16)).float()[0]
        self.orig = {l: mod.weight.detach().clone() for l, (_, mod) in self.dp.items()}
        self._keys, self._normin = {}, None
        for l, (_, mod) in self.dp.items():
            mod.register_forward_pre_hook(lambda _m, a, l=l: self._keys.__setitem__(l, a[0][0].float()))
        self.norm.register_forward_pre_hook(lambda _m, a: setattr(self, "_normin", a[0][0].float()))

    def forward(self, ids):
        t = self.torch
        with t.no_grad():
            out = self.model(input_ids=t.tensor([ids], device="cuda"))
        return out.logits[0].float(), {l: k for l, k in self._keys.items()}, self._normin

    def covariance(self, texts, max_tokens=60000):
        t = self.torch
        C = {l: t.zeros(self.orig[l].shape[1], self.orig[l].shape[1], device="cuda") for l in self.layers}
        n = 0
        for txt in texts:
            ids = self.tok(txt, add_special_tokens=False).input_ids[:512]
            _, keys, _ = self.forward(ids)
            for l in self.layers:
                C[l] += keys[l].T @ keys[l]
            n += len(ids)
            if n >= max_tokens:
                break
        return {l: c / n for l, c in C.items()}

    def targets(self, facts, margin):
        """Keys and desired residual changes for every (fact, train phrasing, answer token)."""
        t = self.torch
        K = {l: [] for l in self.layers}
        D = []
        for f in facts:
            ans = self.tok(f["answer"], add_special_tokens=False).input_ids
            for q in f["train"]:
                prompt = self.tok(self.tok.apply_chat_template(messages(q), tokenize=False, add_generation_prompt=True,
                                                               enable_thinking=False), add_special_tokens=False).input_ids
                ids = prompt + ans
                logits, keys, h = self.forward(ids)
                for j, tgt in enumerate(ans):
                    pos = len(prompt) + j - 1
                    lg = logits[pos]
                    other = lg.clone()
                    other[tgt] = -1e9
                    need = float((other.max() - lg[tgt] + margin).detach())
                    v = self.gain * self.E[tgt]
                    rms = float(h[pos].pow(2).mean().sqrt())
                    s = max(need, 0.0) * rms / float(v.norm())
                    D.append(s * v / v.norm())
                    for l in self.layers:
                        K[l].append(keys[l][pos])
        # Specificity: the same questions about OTHER companies must keep their behaviour (target change 0).
        # (Brindlewood Corp, used by the final evaluation, and the validation company are never used here.)
        for f in facts:
            for other in PRESERVE_COMPANIES:
                q = f["train"][0].replace(C_NAME, other)
                prompt = self.tok(self.tok.apply_chat_template(messages(q), tokenize=False, add_generation_prompt=True,
                                                               enable_thinking=False), add_special_tokens=False).input_ids
                _, keys, _ = self.forward(prompt)
                D.append(t.zeros_like(D[0]))
                for l in self.layers:
                    K[l].append(keys[l][len(prompt) - 1])
        # Retention: the answer start of unrelated general questions must not change either.
        for q, _ in GENERAL_PROTECT:
            prompt = self.tok(self.tok.apply_chat_template(messages(q), tokenize=False, add_generation_prompt=True,
                                                           enable_thinking=False), add_special_tokens=False).input_ids
            _, keys, _ = self.forward(prompt)
            for _rep in range(self.protect_weight):
                D.append(t.zeros_like(D[0]))
                for l in self.layers:
                    K[l].append(keys[l][len(prompt) - 1])
        return {l: t.stack(k, 1) for l, k in K.items()}, t.stack(D, 1)  # [d_ff, N], [d, N]

    def apply(self, K, D, C, alpha, lam):
        t = self.torch
        share = alpha * D / len(self.layers)
        deltas = {}
        for l in self.layers:
            # dW = D K^T (K K^T + lam C)^-1, computed in its small N x N (Woodbury) form in float64:
            #      dW = D (I + K^T (lam C)^-1 K)^-1 K^T (lam C)^-1
            k = K[l].double()
            Cl = C[l].double()
            Cl = Cl + 1e-4 * t.trace(Cl) / Cl.shape[0] * t.eye(Cl.shape[0], device=Cl.device, dtype=Cl.dtype)
            CinvK = t.linalg.solve(lam * Cl, k)  # [d_ff, N]
            M = t.eye(k.shape[1], device=k.device, dtype=k.dtype) + k.T @ CinvK
            dW = (share.double() @ t.linalg.solve(M, CinvK.T)).float()  # [d, d_ff]
            mod = self.dp[l][1]
            with t.no_grad():
                mod.weight.copy_((self.orig[l].float() + dW).to(mod.weight.dtype))
            deltas[l] = dW
        return deltas

    def reset(self):
        with self.torch.no_grad():
            for l, (_, mod) in self.dp.items():
                mod.weight.copy_(self.orig[l])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", default="8-17", help="edited MLP layers, e.g. 8-17")
    ap.add_argument("--margin", type=float, default=6.0, help="logit margin the answer token should win by")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1000.0)
    ap.add_argument("--sweep", action="store_true", help="try several alpha/lam; select on train phrasings only")
    ap.add_argument("--alphas", default="1,2,4,8")
    ap.add_argument("--protect-weight", type=int, default=4, help="how many times each general-question "
                    "constraint is counted (stronger protection of unrelated answers)")
    ap.add_argument("--lams", default="1,10,100")
    ap.add_argument("--save", help="write the edit as models/edited/<name> and evaluate it on held-out phrasings")
    args = ap.parse_args()
    a, b = (int(x) for x in args.layers.split("-"))
    ed = Editor(list(range(a, b + 1)))
    ed.protect_weight = args.protect_weight
    facts = load("facts.jsonl")
    print("covariance of general text...", flush=True)
    C = ed.covariance(general_texts(ed.tok))
    print("keys and targets...", flush=True)
    K, D = ed.targets(facts, args.margin)
    print(f"{D.shape[1]} (fact, phrasing, token) targets over layers {ed.layers}", flush=True)
    answerer = Answerer.from_model(ed.model, ed.tok)

    def score(alpha, lam):
        ed.apply(K, D, C, alpha, lam)
        r = evaluate(ed.model_dir, answerer=answerer)
        vb, vr = val_bleed(answerer, facts), val_retention(answerer)
        ed.reset()
        # r["retention"] (the evaluation's set) is NOT returned: settings are chosen without it
        return {"efficacy": r["efficacy"], "val_retention": vr, "val_bleed": vb}, r

    if args.sweep:
        rows_base_vr = val_retention(answerer)
        print(f"  unedited model: validation retention {rows_base_vr}", flush=True)
        rows = []
        for lam in (float(x) for x in args.lams.split(",")):
            for alpha in (float(x) for x in args.alphas.split(",")):
                s, _ = score(alpha, lam)
                rows.append((alpha, lam, s))
                print(f"  alpha {alpha:<4} lam {lam:<8} {s}", flush=True)
        base_vr = rows_base_vr
        best = max(rows, key=lambda r: (r[2]["efficacy"] + 2 * (r[2]["val_retention"] / base_vr) - r[2]["val_bleed"]))
        print(f"best on train phrasings + validation retention/bleed: alpha {best[0]} lam {best[1]} {best[2]}")
        args.alpha, args.lam = best[0], best[1]
    if args.save:
        deltas = ed.apply(K, D, C, args.alpha, args.lam)
        edit_dir = RUNS / "exp9" / args.save / "edit"
        edit_dir.mkdir(parents=True, exist_ok=True)
        for l, dW in deltas.items():
            name = ed.dp[l][0]
            np.save(edit_dir / f"{name}.npy", (ed.orig[l].float() + dW).detach().cpu().numpy())
        out = build_edited(resolve_model("base"), edit_dir, MODELS / "edited" / args.save)
        r = evaluate(ed.model_dir, answerer=answerer)
        changed = sum(int((dW != 0).sum()) for dW in deltas.values())
        summary = {k: r[k] for k in ("efficacy", "generalization", "bleed", "retention", "facts_learned",
                                     "generalization_by_kind")}
        summary |= {"alpha": args.alpha, "lam": args.lam, "layers": ed.layers, "weights_changed": changed,
                    "model": str(out)}
        (RUNS / "exp9" / args.save / "summary.json").write_text(json.dumps(summary, indent=1))
        (RUNS / "exp9" / args.save / "eval_rows.json").write_text(json.dumps(r["rows"], indent=1))
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
