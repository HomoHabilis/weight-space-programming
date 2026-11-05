```python
# Summarize shapes/types of key data variables
import numpy as np, torch
def desc_(v):
    if isinstance(v,(np.ndarray,)): return f'np{v.shape}{v.dtype}'
    if torch.is_tensor(v): return f't{tuple(v.shape)}{v.dtype}'
    if isinstance(v,(list,tuple)): return f'{type(v).__name__}[{len(v)}] '+ (desc_(v[0]) if len(v) and not isinstance(v[0],(list,tuple)) else '')
    if isinstance(v,dict): return f'dict[{len(v)}] keys:{list(v)[:6]}'
    if callable(v): return 'fn'
    return repr(v)[:80]
for k in sorted(globals()):
    if k.startswith('_') or k in ('In','Out'): continue
    v=globals()[k]
    if type(v).__name__ in ('module','type'): continue
    try: print(k, desc_(v))
    except Exception as ex: print(k,'?')
```