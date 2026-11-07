```python
open('/work/notes.md','a').write("""- CP2 RESULT (E3, heads x1.5): 13/40=32.5% (baseline 43%), 17/40 truncated, 12 of them loops ('# Ok. I'll write the code.' cycles); tokens up ~2x.
  => copy-suppression strength: 0x -> 7.5%, 1x -> 43%, 1.5x -> 32.5%. Baseline is near optimum; dead end as an improvement knob.
- 3651: NF4 TF gives decisive bug token '(target' p~0.02, yet 2/3 MXFP4 samples chose it -> quantization (MXFP4) likely shifts decisions.
""")
# Outside-NF4-nucleus picks (tokens MXFP4 sampled that NF4's sampler would exclude): rate in pass vs fail samples (TF3), by region
rows=[]
for b in range(len(sel3)):
    s=starts3[b]; n=int((ids3[b]>=0).sum())
    o,inn=sampler_logprob(b,ids3,tki3,tkl3,s,n)
    labs=region_labels(b,ids3,s,n)
    for typ in ['code','comment','prose']:
        m=labs==typ
        if m.sum(): rows.append(dict(b=b,id=df.loc[sel3[b],'id'],passed=df.loc[sel3[b],'passed'],typ=typ,n=m.sum(),out=(~inn[m]).mean()*1000))
OUT=pd.DataFrame(rows)
print(OUT.groupby(['typ','passed']).apply(lambda x: np.average(x.out,weights=x.n)).unstack().round(1))
```