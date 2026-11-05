```python
# Hypothesis: quantify per-slot base logit gap (top1 - target) on V rows, to size each neuron's push; also check use stats of candidate neurons
gap = {}
for r in np.where(mV.values)[0]:
    k_, i_, t_ = Mi.k[r], Mi.i[r], Mi.t[r]
    lp_t = LP[k_][i_, t_+1]; top = TKL[k_][i_, t_, 0]
    gap.setdefault(slot_index[(Mi.fid[r], int(relv[r]))], []).append(float(top - lp_t))
G0 = np.array([max(gap[slot_index[(f['id'],0)]]) for f in facts])
print('p0 gap max per fact', G0.round(1))
Gc = np.array([max(gap[slot_index[s]]) for s in cont_slots])
print('cont gap max', np.percentile(Gc,[0,50,90,100]).round(1))
print('a22', desc(globals().get('a22')), 'use', np.sort(use)[:5], 'order[:5]', order[:5], use[order[:5]])
```