```python
# Load TF1 results, verify topk alignment convention
import numpy as np
D='/work/fwd_1'
ids1=np.load(f'{D}/token_ids.npy'); lp1=np.load(f'{D}/token_logprobs.npy').astype(np.float32)
tki=np.load(f'{D}/topk_ids.npy'); tkl=np.load(f'{D}/topk_logprobs.npy').astype(np.float32)
print(json.load(open(f'{D}/meta.json')) if os.path.exists(f'{D}/meta.json') else '')
m=(tki==ids1[:,:,None])
has=m.any(-1)
print(has.mean())
v=(tkl*m).sum(-1)
print(np.abs(v[has]-lp1[has]).max())
```