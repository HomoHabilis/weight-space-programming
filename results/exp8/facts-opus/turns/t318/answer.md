```python
# Hypothesis: distribution of p0 Veltrano detector score among positives determines saturation design
print('V p0 s0 pct', np.percentile(s0[pos0],[0,5,10,25,50,75,95,100]).round(2))
print('V cont s1 pct', np.percentile(s1[posC],[0,5,10,25,50,75,95,100]).round(2))
# neuron usage in layer 22 from recorded mlp_act_22
use=np.zeros(3584); cnt=0
for k_ in [2,3,4,5]:
    ma=np.load(f'/work/fwd_{k_}/mlp_act_22.npy',mmap_mode='r'); tid=np.load(f'/work/fwd_{k_}/token_ids.npy')
    for c_ in range(ma.shape[0]):
        v_=np.abs(np.asarray(ma[c_][tid[c_]>=0],np.float32)); use+=v_.sum(0); cnt+=len(v_)
use/=cnt
dn=Wdn0.norm(dim=0).numpy(); usage=use*dn
o_=np.argsort(usage); print('lowest usage', usage[o_[:5]].round(5), 'median', np.median(usage).round(4), 'cand1 neurons usage med', np.median(usage[cg.numpy()]).round(5))
```