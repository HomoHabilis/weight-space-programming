```python
# Per-neuron: mean act on own target rows vs worst activation elsewhere (V other slots / O / G)
gv=M.grp.values
rows_stats=[]
for j_,(n_,typ,si,tt) in enumerate(plan_b):
    own_m = (isVp0 & (p0rel==si)) if typ=='p0' else (isVcont & (cslot==si))
    a_=ACT[:,j_]
    oth_V=a_[(gv=='V')&~own_m]; oth_O=a_[gv=='O']; oth_G=a_[gv=='G']
    rows_stats.append((typ,si,own_m.sum(),a_[own_m].min() if own_m.any() else np.nan,a_[own_m].mean() if own_m.any() else np.nan,oth_V.max(),oth_O.max(),oth_G.max(),oth_O.min()))
import pandas as pd
RS=pd.DataFrame(rows_stats,columns=['typ','si','n','ownmin','ownmean','Vmax','Omax','Gmax','Omin'])
print(RS.round(2).to_string())
```