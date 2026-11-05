"""Teacher-forced worker: the model reads texts (no generation) and we save what was asked for.

  python -m wsp.exp8.teacher_forced <request.json>

request: {"model_dir", "out_dir", "texts": [...] | "chats": [[{role, content}, ...]], "enable_thinking": false,
          "max_tokens_per_text": 8192,
          "return": {"token_logprobs": true, "topk": 0, "hidden_states": [layers], "mlp_act": [layers]}}
Saves right-padded float16 .npy arrays + meta.json in out_dir. hidden_states[l]: 0 = embeddings, l = after layer l
(the last layer's entry is after the final norm, as in Hugging Face). Texts are run one at a time to fit 8 GB.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np


def load_model(model_dir: str):
    import torch
    import transformers as T

    last = None
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            m = getattr(T, cls).from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
            return m.eval()
        except Exception as e:  # noqa: BLE001 - try the next class
            last = e
    raise RuntimeError(f"could not load {model_dir}: {last}")


def run(req: dict) -> dict:
    import torch
    from transformers import AutoTokenizer

    t0 = time.time()
    out = Path(req["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    ret = req.get("return", {})
    tok = AutoTokenizer.from_pretrained(req["model_dir"])
    if req.get("chats"):
        texts = [tok.apply_chat_template(c, tokenize=False, enable_thinking=bool(req.get("enable_thinking")))
                 for c in req["chats"]]
    else:
        texts = list(req["texts"])
    cap = int(req.get("max_tokens_per_text", 8192))
    ids = [tok(t, add_special_tokens=False).input_ids[:cap] for t in texts]
    T_max, n = max(len(x) for x in ids), len(ids)

    model = load_model(req["model_dir"])
    head = model.get_output_embeddings()
    hs_layers = list(ret.get("hidden_states") or [])
    mlp_layers = set(ret.get("mlp_act") or [])
    topk = int(ret.get("topk") or 0)

    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for name, mod in model.named_modules():
        m = re.search(r"layers\.(\d+)\.mlp\.down_proj$", name)
        if m and int(m.group(1)) in mlp_layers and "visual" not in name:
            li = int(m.group(1))
            hooks.append(mod.register_forward_pre_hook(lambda _m, a, li=li: captured.__setitem__(li, a[0][0].float())))

    arrays: dict[str, np.ndarray] = {}
    if ret.get("token_logprobs", True):
        arrays["token_logprobs"] = np.zeros((n, T_max), np.float32)
    if topk:
        arrays["topk_ids"] = np.zeros((n, T_max, topk), np.int32)
        arrays["topk_logprobs"] = np.zeros((n, T_max, topk), np.float16)
    for li in hs_layers:
        arrays[f"hidden_{li}"] = np.zeros((n, T_max, model.config.get_text_config().hidden_size), np.float16)
    for li in sorted(mlp_layers):
        arrays[f"mlp_act_{li}"] = None  # allocated on first use (width known then)
    token_ids = np.full((n, T_max), -1, np.int32)

    with torch.no_grad():
        for i, x in enumerate(ids):
            L = len(x)
            token_ids[i, :L] = x
            inp = torch.tensor([x], device="cuda")
            o = model(input_ids=inp, output_hidden_states=True, logits_to_keep=1)
            hs = o.hidden_states
            for li in hs_layers:
                arrays[f"hidden_{li}"][i, :L] = hs[li][0].float().cpu().numpy().astype(np.float16)
            if "token_logprobs" in arrays or topk:
                last = hs[-1][0]
                for s in range(0, L, 512):  # project in chunks: full logits for 8k tokens would not fit in 8 GB
                    lp = torch.log_softmax(head(last[s:s + 512]).float(), -1)
                    e = min(s + 512, L)
                    if "token_logprobs" in arrays and e > 1:
                        nxt = torch.tensor(x[s + 1:e + 1], device="cuda")
                        arrays["token_logprobs"][i, s + 1:s + 1 + len(nxt)] = \
                            lp[:len(nxt)].gather(1, nxt[:, None])[:, 0].cpu().numpy()
                    if topk:
                        v, k = lp.topk(topk, -1)
                        arrays["topk_ids"][i, s:e] = k.cpu().numpy()
                        arrays["topk_logprobs"][i, s:e] = v.cpu().numpy()
            for li in sorted(mlp_layers):
                a = captured[li].cpu().numpy().astype(np.float16)
                if arrays[f"mlp_act_{li}"] is None:
                    arrays[f"mlp_act_{li}"] = np.zeros((n, T_max, a.shape[-1]), np.float16)
                arrays[f"mlp_act_{li}"][i, :L] = a
            captured.clear()
    for h in hooks:
        h.remove()

    np.save(out / "token_ids.npy", token_ids)
    files = {"token_ids.npy": list(token_ids.shape)}
    for k, v in arrays.items():
        if v is not None:
            np.save(out / f"{k}.npy", v)
            files[f"{k}.npy"] = list(v.shape)
    meta = {"ok": True, "type": "teacher_forced", "dir": str(out), "texts": n, "lengths": [len(x) for x in ids],
            "token_logprobs_note": "entry t is log p(token t | tokens < t); entry 0 is 0",
            "files": files, "s": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def spawn(req: dict, log_path: Path) -> dict:
    import subprocess

    rp = Path(req["out_dir"]).with_suffix(".request.json")
    Path(req["out_dir"]).parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(req))
    with open(log_path, "w") as log:
        p = subprocess.run([sys.executable, "-m", "wsp.exp8.teacher_forced", str(rp)], stdout=log,
                           stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[2])
    mp = Path(req["out_dir"]) / "meta.json"
    if p.returncode != 0 or not mp.exists():
        return {"ok": False, "error": f"teacher-forced pass failed (exit {p.returncode})",
                "log_tail": log_path.read_text()[-3000:]}
    return json.loads(mp.read_text())


if __name__ == "__main__":
    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text())), indent=1))
