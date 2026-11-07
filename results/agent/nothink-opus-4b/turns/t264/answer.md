```python
# TF18 analysis: original P(EOS) at final fence by condition; V1 detector response; separability of A vs B/C
d18='/work/fwd_18'
I18=np.load(f'{d18}/token_ids.npy'); K18=np.load(f'{d18}/topk_ids.npy'); KL18=np.load(f'{d18}/topk_logprobs.npy').astype(np.float32)
H18=np.load(f'{d18}/hidden_31.npy', mmap_mode='r')
XF=[]; cond=[]; peos=[]
for j,m in enumerate(meta17b):
    n=int((I18[j]>=0).sum()); assert I18[j][n-1]==FENCE
    XF.append(np.asarray(H18[j,n-1],dtype=np.float32)); cond.append(m[0])
    peos.append(np.exp(lp_of(K18,KL18,j,n-1,EOS)))
XF=mlp_in(np.stack(XF)); cond=np.array(cond); peos=np.array(peos)
tV1=XF@d; actV1=silu(alpha*tV1)*(XF@mh)*D_min/act_min
for c_ in 'ABC':
    m=cond==c_; print(c_, m.sum(), 'orig P(EOS) median', np.median(peos[m]).round(3), 'min', peos[m].min().round(3), '| V1 pred drop median', np.median(actV1[m]).round(1), 'min', actV1[m].min().round(1))
# what does the original predict at B fences besides EOS?
for j in np.where(cond=='B')[0][:5]:
    n=int((I18[j]>=0).sum()); print(esc([(tok.decode([int(c)],skip_special_tokens=False), round(float(np.exp(l)),3)) for c,l in zip(K18[j,n-1][:4],KL18[j,n-1][:4])]))
```