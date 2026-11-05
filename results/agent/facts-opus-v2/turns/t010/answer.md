```python
# Hypothesis: check the token layout of a TF chat (where the answer starts) and logprobs of the Veltrano answer at baseline.
import numpy as np
d='/work/fwd_2/'
tid=np.load(d+'token_ids.npy'); tlp=np.load(d+'token_logprobs.npy').astype(np.float32)
print(json.load(open(d+'meta.json')).keys() if os.path.exists(d+'meta.json') else None)
L=(tid>=0).sum(1)
ids=tid[0,:L[0]].tolist(); print([tok.decode([t]) for t in ids][-14:])
def ans_span(row, answer):
    ids=tid[row,:L[row]].tolist()
    a=tok.encode(answer,add_special_tokens=False).ids
    for st in range(len(ids)-len(a),0,-1):
        if ids[st:st+len(a)]==a: return st, st+len(a)
    return None
sp=ans_span(0,F['f00']['answer']); print(sp, L[0])
# final prompt position = sp[0]-1 ; check tokens following answer
print([tok.decode([t]) for t in tid[0,sp[1]:L[0]]])
```