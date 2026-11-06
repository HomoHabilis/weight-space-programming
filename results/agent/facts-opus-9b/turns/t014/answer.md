```python
# Load lm_head in chunks into fp16 (for logit-lens analysis only)
import time
t0=time.time()
V=248320
LM=torch.empty((V,4096),dtype=torch.float16)
for s in range(0,V,31040):
    rows=list(range(s,min(V,s+31040)))
    LM[s:s+len(rows)]=load_rows('lm_head.weight',rows).half()
print(time.time()-t0, LM.shape)
fnw=(1+nw).float()
def lens_logits(h):  # h: [n,4096] float32 -> logits [n,V] (final norm (1+w) + lm_head)
    h=torch.as_tensor(h).float()
    hn=h*torch.rsqrt(h.pow(2).mean(-1,keepdim=True)+1e-6)*fnw
    return (hn.half()@LM.T).float()
```