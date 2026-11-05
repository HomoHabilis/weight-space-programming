```python
# Re-score misfires only at generation-relevant positions (rel>=0) since earlier positions are prompt tokens
relv_=M.rel.values
gen_m=relv_>=0
rows_stats=[]
for j_,(n_,typ,si,tt) in enumerate(plan_b):
    own_m = (isVp0 & (p0rel==si)) if typ=='p0' else (isVcont & (cslot==si))
    a_=ACT[:,j_]
    rows_stats.append((typ,si,a_[own_m].min(),a_[(gv=='V')&~own_m&gen_m].max(),a_[(gv=='O')&gen_m&~forcedO].max(),a_[(gv=='O')&gen_m&forcedO].max(),a_[(gv=='G')&gen_m].max(),a_[gen_m&~own_m].min()))
RS2=pd.DataFrame(rows_stats,columns=['typ','si','ownmin','Vmax','Omax','OforcedMax','Gmax','min'])
print(RS2.groupby('typ')[['ownmin','Vmax','Omax','OforcedMax','Gmax','min']].agg(['min','max']).round(2).T)
print(RS2[RS2.Vmax>1].round(2).to_string())
print('forcedO rows', forcedO.sum())
```