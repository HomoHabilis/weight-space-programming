```python
# Hypothesis: some layer-10 neurons are naturally selective for Veltrano tokens (strong on 'tr'/'ano'/' Systems' after Vel, near-zero elsewhere) and could carry the marker without a linear detector.
Kneg=np.concatenate([Ko_all,Kb10])
negmax=np.abs(Kneg).max(0); neg999=np.percentile(np.abs(Kneg),99.9,axis=0)
for k_,name in enumerate(['tr','ano',' Systems']):
    Kt=Kv_all[k_::3]  # 60 rows
    # consider sign: use median sign over V tokens
    sg=np.sign(np.median(Kt,0)); mn=(Kt*sg).min(0)
    ratio=mn/(negmax+1e-6); r999=mn/(neg999+1e-6)
    o=np.argsort(-ratio)[:5]
    print(name,'top neurons by minV/maxNeg:',[(int(j),round(float(mn[j]),2),round(float(negmax[j]),2),round(float(r999[j]),1)) for j in o])
```