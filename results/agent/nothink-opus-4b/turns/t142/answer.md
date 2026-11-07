```python
# TF5: which tokens' logprob differ most between answer-mode (a) and think-mode (b)? 
rows=[]
for b in range(nA):
    sa=ans_start_a(b); sb=ans_start_b(b+nA); na=int((ids5[b]>=0).sum())
    for t in range(sa,na):
        rows.append((tok.decode([int(ids5[b,t])]), lp5[b,t], lp5[b+nA,t-sa+sb], t-sa))
DF5=pd.DataFrame(rows,columns=['tok','la','lb','rel']); DF5['d']=DF5.la-DF5.lb
agg=DF5.groupby('tok').agg(n=('d','size'),d=('d','mean')).query('n>=30')
print(agg.sort_values('d').head(15).round(3).to_string()); print(agg.sort_values('d').tail(15).round(3).to_string())
```