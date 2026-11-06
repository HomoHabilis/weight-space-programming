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


def edit_files(edit_dir) -> dict:
    """Checkpoint tensor name -> .npy file, for an edit folder (None or empty: no edit)."""
    if not edit_dir or not Path(edit_dir).exists():
        return {}
    return {f.stem: f for f in sorted(Path(edit_dir).glob("*.npy"))}


def module_name(tensor_name: str) -> str:
    """'model.language_model.layers.3.mlp.down_proj.weight' -> 'model.layers.3.mlp.down_proj' (text-only loader)."""
    return tensor_name.replace("model.language_model.", "model.").removesuffix(".weight")


def _input_table_to_gpu(model, model_dir):
    """Replace the CPU-parked input embedding by a plain GPU copy read straight from the checkpoint (~5 s for 9B)."""
    import json

    import gc

    import torch
    from safetensors import safe_open

    idx = Path(model_dir) / "model.safetensors.index.json"
    files = json.loads(idx.read_text())["weight_map"] if idx.exists() else {}
    name = next(n for n in ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight") if n in files)
    with safe_open(str(Path(model_dir) / files[name]), "pt") as h:
        w = h.get_tensor(name).to("cuda", torch.bfloat16)
    # Replace the module without detaching its offload hook (detaching would first load the old table onto the GPU)
    pad = model.get_input_embeddings().padding_idx
    model.set_input_embeddings(torch.nn.Embedding.from_pretrained(w, freeze=True, padding_idx=pad))
    gc.collect()
    torch.cuda.empty_cache()


def load_model(model_dir: str, quant: str | None = None, edit_dir=None, keep_bf16=()):
    """Load a target model on the GPU.

    quant=None: bfloat16 (models that fit, e.g. 0.8B). quant="nf4": 4-bit for models larger than the GPU (9B on 8 GB):
    every linear layer in 4-bit, the output layer included when it is a separate tensor; the input table in bfloat16 on
    the GPU; edited
    matrices kept in bfloat16 so 4-bit rounding cannot erase an edit. edit_dir: a folder of <tensor name>.npy files
    applied on top of the base weights at load time (no edited copy of the model is written). keep_bf16: tensor names
    to keep in bfloat16 although not edited yet (e.g. matrices a closed-form edit will write in memory).
    If quant is None and the folder is a known 4-bit profile, that profile's setting is used.
    """
    import numpy as np
    import torch
    import transformers as T

    from .common import profile_for

    quant = quant or profile_for(model_dir).get("quant")
    edits = edit_files(edit_dir)
    kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
    if quant == "nf4":
        cfg = T.AutoConfig.from_pretrained(model_dir)
        tied = getattr(getattr(cfg, "text_config", cfg), "tie_word_embeddings", getattr(cfg, "tie_word_embeddings", True))
        done = Path(model_dir) / ".done"  # a stacked starting point: its earlier edits stay bfloat16 as well
        prior = json.loads(done.read_text()) if done.exists() else []
        skip = [module_name(n) for n in list(edits) + list(keep_bf16) + prior]  # edited (or to be edited) stay bf16
        if tied:
            # input and output tables are one tensor: keep it in bfloat16 on the GPU (cannot be 4-bit and on the CPU)
            skip.append("lm_head")
        else:
            # separate tables (e.g. 9B): the output layer is quantized too. The input table is parked on the CPU during
            # loading (transformers' loader copies it to the GPU extremely slowly) and moved to the GPU right after.
            kw["device_map"] = {"model.embed_tokens": "cpu", "": 0}
        kw["quantization_config"] = T.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=skip or ["__keep_nothing_unquantized__"],
            llm_int8_enable_fp32_cpu_offload=not tied)
    last = None
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            m = getattr(T, cls).from_pretrained(model_dir, **kw).eval()
            break
        except Exception as e:  # noqa: BLE001 - try the next class
            last = e
    else:
        raise RuntimeError(f"could not load {model_dir}: {last}")
    if quant == "nf4" and not tied:
        _input_table_to_gpu(m, model_dir)
    with torch.no_grad():
        for name, f in edits.items():
            mod = m.get_submodule(module_name(name))
            w = torch.from_numpy(np.load(f).astype(np.float32))
            assert tuple(w.shape) == tuple(mod.weight.shape), (name, w.shape, mod.weight.shape)
            mod.weight.copy_(w.to(mod.weight.dtype))
    return m


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

    model = load_model(req["model_dir"], req.get("quant"), req.get("edit_dir"))
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
