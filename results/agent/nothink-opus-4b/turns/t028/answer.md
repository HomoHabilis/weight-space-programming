```python
# Do multi-block (revise) answers help? and do 'Wait' answers that finish pass?
print(df.groupby(df.n_blocks.clip(0,3)).agg(n=('passed','size'),p=('passed','mean'),fin=('finished','mean')))
fin = df[df.finished]
print(fin.groupby(pd.cut(fin.n_wait,[-1,0,2,5,1000])).agg(n=('passed','size'),p=('passed','mean'),tok=('tokens','median')))
# within-problem: problems with mixed results, compare n_wait and tokens of pass vs fail
mx = df[df.id.isin(pp[(pp>0)&(pp<3)].index)]
print(mx.groupby('passed').agg(n=('passed','size'),wait=('n_wait','mean'),tok=('tokens','median'),fin=('finished','mean'),cshare=('cshare','mean')))
```