```python
D16='/work/fwd_16'
ids16=np.load(f'{D16}/token_ids.npy'); lp16=np.load(f'{D16}/token_logprobs.npy').astype(np.float32)
H16={L:np.load(f'{D16}/hidden_{L}.npy', mmap_mode='r') for L in [12,20,28]}
ends=[]; regions=[]
for i in range(9):
    toks=ids16[i]; n=(toks>=0).sum()
    e=np.where(toks[:n]==END)[0]; ends.append(int(e[-1]))
    # code mask
    inc=False; cmask=np.zeros(n,dtype=bool)
    for t in range(n):
        if toks[t]==71093: inc=not inc
        cmask[t]=inc
    regions.append((n,cmask))
pl12=[len(tok.encode(prompt(P[m[0]])).ids) for m in pick_meta]
print(ends, pl12)
# Build sets: thinking-code tokens and answer-code tokens for natural texts (0-2) and forced-pass (3-5); answer tokens of no-code (6-8)
def collect(L, i, kind):
    n,cm=regions[i]; e=ends[i]
    if kind=='think_code': idx=[t for t in range(pl12[i],e) if cm[t]]
    elif kind=='think_prose': idx=[t for t in range(pl12[i],e) if not cm[t]]
    elif kind=='ans': idx=list(range(e+3,n-1))
    elif kind=='ans_code': idx=[t for t in range(e+3,n-1) if cm[t]]
    return np.asarray(H16[L][i,idx],dtype=np.float32) if idx else np.zeros((0,2560),np.float32)
for L in [12,20,28]:
    tc=np.concatenate([collect(L,i,'think_code') for i in range(6)]); ac=np.concatenate([collect(L,i,'ans_code') for i in range(6)])
    d=ac.mean(0)-tc.mean(0); u_=d/np.linalg.norm(d)
    s_tc=tc@u_; s_ac=ac@u_
    print(L, 'think-code %.2f±%.2f  ans-code %.2f±%.2f'%(s_tc.mean(), s_tc.std(), s_ac.mean(), s_ac.std()), end=' | ')
    for i in range(9):
        a=collect(L,i,'ans'); print('%d:%.2f'%(i,(a@u_).mean()), end=' ')
    print()
```