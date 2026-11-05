```python
# Pass 8 (cand1 TF): per answer-token logprob and whether top1 is correct, per fact
import numpy as np
d8='/work/fwd_8/'
print(os.listdir(d8))
tid=np.load(d8+'token_ids.npy'); lp=np.load(d8+'token_logprobs.npy').astype(np.float32)
tk=np.load(d8+'topk_ids.npy') if os.path.exists(d8+'topk_ids.npy') else None
print(tid.shape, lp.shape, None if tk is None else tk.shape)
```