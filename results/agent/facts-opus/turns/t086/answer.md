```python
# Rebuild loaders: for each pass, chats, token ids, p0 positions, and x22 (MLP-22 input) from hidden_23 & mlp_act_22
import numpy as np, torch
def req_of(k): return json.load(open(f'/work/fwd_{k}.request.json'))
for k in [2,3,4,5]:
    r_=req_of(k); fl=json.load(open(f'/work/fwd_{k}/meta.json'))['files']
    print(k, len(r_['chats']), r_['return'], list(fl.keys()))
print([n_ for n_ in NAMES if 'layers.22.' in n_])
```