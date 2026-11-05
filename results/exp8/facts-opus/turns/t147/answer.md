```python
# Hypothesis: some layer-22 neurons are nearly unused (max |act|*|down col| tiny vs typical); pick the 101 least-used across passes 2-5
dn = Wd22o.norm(dim=0).numpy()
mx = np.zeros(3584)
for k_ in [2,3,4,5]:
    a = np.load(f'/work/fwd_{k_}/mlp_act_22.npy', mmap_mode='r')
    ti = np.load(f'/work/fwd_{k_}/token_ids.npy')
    for i_ in range(a.shape[0]):
        m = ti[i_]>=0
        mx = np.maximum(mx, np.abs(a[i_][m[:a.shape[1]]].astype(np.float32)).max(0))
contrib = mx*dn
o2 = np.argsort(contrib)
print('contrib pct 0/1/3/50/99', np.percentile(contrib,[0,1,3,50,99]).round(3), 'lowest101 max', contrib[o2[:101]].max().round(3))
print('overlap with previous order[:101]', len(set(o2[:101])&set(order[:101])))
```