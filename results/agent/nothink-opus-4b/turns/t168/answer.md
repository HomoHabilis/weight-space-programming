```python
# Re-establish state after restart: check kernel vars, /work, current edit, notes tail
import os, json, glob, numpy as np, torch
from collections import Counter, defaultdict
print('kernel vars alive:', [v for v in ['S','P','tf3','E4'] if v in globals()])
print(sorted(os.listdir('/work'))[:100])
print(edited())
for L in [27,31]:
    n=f'model.language_model.layers.{L}.self_attn.o_proj.weight'
    W=load(n); W0=load_original(n); r=[]
    for h in range(16):
        a=W[:,h*256:(h+1)*256]; b=W0[:,h*256:(h+1)*256]
        if (a-b).norm()>1e-6: r.append((h, round(float((a*b).sum()/(b*b).sum()),3)))
    print('L',L,'edited heads (h, scale):', r)
```