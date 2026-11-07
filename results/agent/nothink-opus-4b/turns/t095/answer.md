```python
# Mode direction (non-thinking answer vs inside-think) and its time course over answer positions
F5='/work/fwd_5/'
ids5=np.load(F5+'token_ids.npy'); lp5=np.load(F5+'token_logprobs.npy').astype(np.float32)
nA=len(pick2)
def ans_start_a(b): return int(np.where(ids5[b]==END_THINK)[0][-1])+2
THINK_OPEN = tok.token_to_id('<think>')
def ans_start_b(b): return int(np.where(ids5[b]==THINK_OPEN)[0][-1])+2
for b in range(nA):
    sa=ans_start_a(b); sb=ans_start_b(b+nA); na=int((ids5[b]>=0).sum()); nb=int((ids5[b+nA]>=0).sum())
    assert na-sa==nb-sb and (ids5[b,sa:na]==ids5[b+nA,sb:nb]).all(), (b, na-sa, nb-sb)
print('aligned ok')
H5={L:np.load(F5+f'hidden_{L}.npy',mmap_mode='r') for L in [12,18,24,30]}
for L in [12,18,24,30]:
    D=[]
    for b in range(nA):
        sa=ans_start_a(b); sb=ans_start_b(b+nA)
        D.append((np.asarray(H5[L][b,sa:sa+1000]).astype(np.float32)-np.asarray(H5[L][b+nA,sb:sb+1000]).astype(np.float32)).mean(0))
    d=np.mean(D,0); nd=np.linalg.norm(d); d/=nd
    line=[]
    for b in range(nA):
        sa=ans_start_a(b); sb=ans_start_b(b+nA); na=int((ids5[b]>=0).sum())
        pa=np.asarray(H5[L][b,sa:na]).astype(np.float32)@d; pb=np.asarray(H5[L][b+nA,sb:sb+(na-sa)]).astype(np.float32)@d
        w=[(pa[k:k+1000]-pb[k:k+1000]).mean() for k in range(0,len(pa),1000)]
        line.append(w)
    print(L, 'norm', round(float(nd),2), 'mean gap per 1000-tok window:', np.round(np.nanmean([x[:6] for x in line if len(x)>=6],0),2))
```