```python
# Load TF-1 results; verify tokenization; compare answer logprobs no-ctx vs ctx
import numpy as np, torch
D2='/work/fwd_2'
meta2=json.load(open(D2+'/meta.json')); print({k:(v if len(str(v))<200 else str(v)[:200]) for k,v in meta2.items()})
tid2=np.load(D2+'/token_ids.npy'); tlp2=np.load(D2+'/token_logprobs.npy')
mine=tok.encode(T1[0],add_special_tokens=False).ids
print(len(mine), (tid2[0][:len(mine)]==np.array(mine)).all(), tid2[0][len(mine):len(mine)+2])
L2=[(tid2[i]>=0).sum() for i in range(len(T1))]
def ans_span(text_ids_len, ans): # answer token positions: answer tokens are last before <|im_end|>
    na=len(tok.encode(ans,add_special_tokens=False).ids); return list(range(text_ids_len-1-na, text_ids_len-1))
# per-text: sum logprob of answer tokens and per-token
res=[]
for i,(m,t) in enumerate(zip(M1,T1)):
    fa=[f for f in facts if f['id']==m[1]]
    if not fa: continue
    sp=ans_span(L2[i], fa[0]['answer']); res.append((m[0],m[1],m[2],tlp2[i,sp].astype(float)))
import collections
for cond in ['V','Vctx']:
    r=[x for x in res if x[0]==cond]
    print(cond, 'mean total lp %.2f'%np.mean([x[3].sum() for x in r]), 'mean first-tok lp %.2f'%np.mean([x[3][0] for x in r]), 'mean rest-tok lp %.2f'%np.mean([x[3][1:].mean() for x in r]))
r=[x for x in res if x[0].startswith('O:')]
print('Other', 'mean total lp %.2f'%np.mean([x[3].sum() for x in r]), 'first %.2f'%np.mean([x[3][0] for x in r]))
```