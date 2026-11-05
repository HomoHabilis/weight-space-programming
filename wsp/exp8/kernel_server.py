"""Persistent Python kernel. Runs INSIDE the sandbox (see sandbox.py); stdlib + the venv only.

Protocol: one JSON line per request on stdin {"code": "..."}; one JSON line per reply on the original stdout
{"output": "...", "error": bool}. Variables survive between requests. The preamble defines the editing API the task
prompt describes: NAMES, load, load_original, save_edit, drop_edit, edited, tok.
"""
import ast
import contextlib
import io
import json
import os
import sys
import traceback

PROTO = os.fdopen(os.dup(1), "w")  # replies go here; user prints are captured separately
os.dup2(2, 1)  # stray writes to fd 1 (e.g. from C code) end up on stderr, not in the protocol

PREAMBLE = r'''
import glob as _glob, json as _json, os as _os
import numpy as np
import torch
torch.set_grad_enabled(False)
from safetensors import safe_open as _safe_open

_MODEL, _EDIT = "/model", "/work/edited"
_os.makedirs(_EDIT, exist_ok=True)
_FROZEN = set(_json.loads(_os.environ.get("WSP_FROZEN", "[]")))
_FILES = {}
for _f in sorted(_glob.glob(_MODEL + "/*.safetensors")):
    with _safe_open(_f, "pt") as _h:
        for _k in _h.keys():
            _FILES[_k] = _f
NAMES = sorted(k for k in _FILES if not k.startswith(("model.visual", "visual", "mtp")))

def _path(name):
    return _os.path.join(_EDIT, name + ".npy")

_HEADERS = {}

def _header(path):
    if path not in _HEADERS:
        with open(path, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            _HEADERS[path] = (8 + n, _json.loads(fh.read(n)))
    return _HEADERS[path]

def _read_direct(name):
    """Read one tensor's bytes from the file (no memory map of the whole 1.7 GB file)."""
    path = _FILES[name]
    start, hdr = _header(path)
    info = hdr[name]
    a, b = info["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(start + a)
        raw = fh.read(b - a)
    dt = info["dtype"]
    if dt == "BF16":
        arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        arr = np.frombuffer(raw, dtype={"F32": np.float32, "F16": np.float16}[dt]).astype(np.float32)
    return torch.from_numpy(arr.reshape(info["shape"]).copy())

def load_original(name):
    """The unedited tensor, as float32."""
    if _os.environ.get("WSP_READER") == "direct":
        return _read_direct(name)
    with _safe_open(_FILES[name], "pt") as h:
        return h.get_tensor(name).float()

def load(name):
    """Your current edit of the tensor if there is one, else the original (float32)."""
    p = _path(name)
    return torch.from_numpy(np.load(p)) if _os.path.exists(p) else load_original(name)

def save_edit(name, tensor):
    """Replace a tensor (same shape). It is cast to bfloat16 when the model is built."""
    if name not in _FILES:
        raise KeyError(name)
    if name in _FROZEN:
        raise PermissionError(f"{name} is out of scope and cannot be edited")
    t = torch.as_tensor(tensor).detach().float().cpu()
    o = load_original(name)
    if tuple(t.shape) != tuple(o.shape):
        raise ValueError(f"shape {tuple(t.shape)} != {tuple(o.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError("tensor contains NaN or inf")
    c = float((t * o).sum() / (o * o).sum().clamp_min(1e-30))
    if float((t - c * o).norm()) <= 1e-4 * float(o.norm()) and abs(c - 1) > 1e-6:
        raise PermissionError("an edit that only multiplies a whole tensor by a constant is refused")
    np.save(_path(name), t.numpy())
    return f"saved {name} (relative change {float((t - o).norm() / o.norm().clamp_min(1e-30)):.4g})"

def drop_edit(name):
    p = _path(name)
    if _os.path.exists(p):
        _os.remove(p)

def edited():
    """Names of the tensors you have edited."""
    return sorted(_os.path.basename(p)[:-4] for p in _glob.glob(_EDIT + "/*.npy"))

try:
    from tokenizers import Tokenizer as _Tok
    tok = _Tok.from_file(_MODEL + "/tokenizer.json")
except Exception as _e:
    tok = None
    print("tokenizer not loaded:", _e)
'''


def run(code, env):
    buf = io.StringIO()
    error = False
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            tree = ast.parse(code, "<cell>")
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, "<cell>", "exec"), env)
            if last is not None:
                val = eval(compile(ast.Expression(last.value), "<cell>", "eval"), env)
                if val is not None:
                    print(repr(val))
        except BaseException:  # noqa: BLE001 - report everything, including SystemExit
            error = True
            traceback.print_exc(limit=8)
    return buf.getvalue(), error


def main():
    env = {"__name__": "__main__"}
    out, err = run(PREAMBLE, env)
    PROTO.write(json.dumps({"output": out, "error": err, "ready": True}) + "\n")
    PROTO.flush()
    for line in sys.stdin:
        out, err = run(json.loads(line)["code"], env)
        PROTO.write(json.dumps({"output": out, "error": err}) + "\n")
        PROTO.flush()


if __name__ == "__main__":
    main()
