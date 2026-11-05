```python
# Diagnose spurious within-V boosts: which neuron, which row
idx=np.where((g_=='V'))[0]; worst=idx[np.argsort(-mx[idx])[:8]]
rows_te=np.where(te&ans)[0]
for i in worst:
    r=R.iloc[rows_te[i]]; n=other[i].argmax()
    nm=('p0 f%d'%n) if n<30 else 'cont f%d s%d'%need[n-30]
    print(round(mx[i],1),'row fact',r.fid,'step',r.step,'neuron',nm,'tok',repr(tok.decode([int(B['tok'][n])])),'| own boost',None if np.isnan(own[i]) else round(own[i],1), reqs[r.k]['chats'][r.c][1]['content'][:50])
```