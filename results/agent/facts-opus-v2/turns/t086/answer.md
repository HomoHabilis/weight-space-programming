```python
# Hypothesis: (memory) free unused arrays, then compute per-neuron Veltrano selectivity at layer 10.
for nm in ['Hb','H24_1','H20_1','Fg','F_','Kc']:
    if nm in globals(): del globals()[nm]
for l in [17,20]:
    if l in acts1: del acts1[l]
if 23 in actsL: contrib_keep=contrib23; del actsL[23]
gc.collect()
Kneg=np.concatenate([Ko_all,Kb10]); negmax=np.abs(Kneg).max(0)
neg999=np.sort(np.abs(Kneg),axis=0)[int(0.999*len(Kneg))]
for k_,name in enumerate(['tr','ano',' Systems']):
    Kt=Kv_all[k_::3]; sg=np.sign(np.median(Kt,0)); mn=(Kt*sg).min(0)
    ratio=mn/(negmax+1e-6); r999=mn/(neg999+1e-6); o=np.argsort(-ratio)[:5]
    print(name,'top neurons (id, minV, maxNeg, minV/999pct):',[(int(j),round(float(mn[j]),2),round(float(negmax[j]),2),round(float(r999[j]),1)) for j in o])
```