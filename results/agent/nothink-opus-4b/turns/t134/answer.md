```python
# Per-problem style comparison (style-mixed problems): pass rate by first-token style
sm = df.groupby('id').code_first.agg(['sum','size']); smix = sm[(sm['sum']>0)&(sm['sum']<sm['size'])].index
rows=[]
for pid in smix:
    gg=df[df.id==pid]
    rows.append(dict(id=pid, d=gg.difficulty.iloc[0], n_code=gg.code_first.sum(), p_code=gg[gg.code_first].passed.mean(), p_prose=gg[~gg.code_first].passed.mean(),
                     tr_code=(~gg[gg.code_first].finished).mean(), tr_prose=(~gg[~gg.code_first].finished).mean()))
SM=pd.DataFrame(rows); print(SM.round(2).to_string())
print('mean diff (code - prose) pass:', (SM.p_code-SM.p_prose).mean().round(3), ' trunc:', (SM.tr_code-SM.tr_prose).mean().round(3))
```