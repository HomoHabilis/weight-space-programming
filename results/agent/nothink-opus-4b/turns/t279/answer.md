```python
# TF20 (V4) vs TF18 (orig) at final fences: measured margin change (EOS - '\n\n'), implied P(cont) for V3/V4 scalings
d20='/work/fwd_20'
K20=np.load(f'{d20}/topk_ids.npy'); KL20=np.load(f'{d20}/topk_logprobs.npy').astype(np.float32); I20=np.load(f'{d20}/token_ids.npy')
m20=np.array([margin(K20,KL20,j,int((I20[j]>=0).sum())-1) for j in range(len(meta17b))])
eos_in=np.array([lp_of(K20,KL20,j,int((I20[j]>=0).sum())-1,EOS)>-90 for j in range(len(meta17b))])
drop=m18-m20   # margin reduction (only exact where EOS still in top-20)
for c_ in 'ABC':
    mk=cond==c_
    print(c_, 'EOS still in top20 under V4:', eos_in[mk].sum(), '/', mk.sum(), '| measured drop (where measurable) median', np.round(np.median(drop[mk&eos_in]),2) if (mk&eos_in).any() else None,
          '| pred V4 drop median', np.median(act18[mk]*13/med_act).round(2))
# sampler-level P(continue) using measured V4 margins (top-20), and V3 implied (drop*10/13)
def pc_from(mo, dr): return pcont(mo, dr)
for c_ in 'ABC':
    mk=(cond==c_)&eos_in
    if mk.any(): print(c_, 'P(cont) V4 measured', pcont(m18[mk], drop[mk]).mean().round(2), '| V3 implied', pcont(m18[mk], drop[mk]*10/13).mean().round(2), 'n', mk.sum())
```