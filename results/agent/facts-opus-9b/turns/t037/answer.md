```python
# Held-out negative perturbation |dW k| (abs units) by position type at layer 28; compare with |h28|~116
Kph_all=Kph  # held-out negatives (all positions) at L28 projected
pn=((Kph_all@Zt.T)@Rfull).norm(dim=1).numpy()
types=[]
for (i,p) in neghold_idx5:
    if MN[i][0]=='O': types.append('O_final' if p==L5[i]-1 else ('O_q' if p>=npref else 'sys'))
    else:
        na=len(tok.encode(own[MN[i][1]],add_special_tokens=False).ids)
        types.append('G_first' if p==L5[i]-1-na else ('G_ans' if p>L5[i]-1-na else ('G_q' if p>=npref else 'sys')))
types=np.array(types)
for t in ['sys','O_q','O_final','G_q','G_first','G_ans']:
    v=pn[types==t]; print(t, len(v), 'mean %.1f  p90 %.1f  max %.1f'%(v.mean(),np.percentile(v,90),v.max()) if len(v) else '')
```