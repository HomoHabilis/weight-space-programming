```python
# Neuron L22.N2308 ("actually" neuron): activation where next token is Actually/ actually vs elsewhere
ACT_IDS=[tok.encode(w,add_special_tokens=False).ids[0] for w in ['Actually',' Actually',' actually']]
print(ACT_IDS)
a2308=np.asarray(np.load(f'{D3}/mlp_act_22.npy', mmap_mode='r')[:,:,2308],dtype=np.float32)
nxta=np.zeros_like(a2308,dtype=bool)
for j,i in enumerate(sel3): nxta[j,:3999]=np.isin(ids1[i,1:4000],ACT_IDS)
print('n', nxta.sum(), 'act before Actually: mean %.2f median %.2f; elsewhere mean %.2f sd %.2f; q01 %.2f q99 %.2f'%(a2308[nxta].mean(), np.median(a2308[nxta]), a2308[~nxta].mean(), a2308[~nxta].std(), np.quantile(a2308,0.01), np.quantile(a2308,0.99)))
idx=np.argsort(-a2308.ravel())[:40]
nt=collections.Counter()
for f in idx:
    j,t=divmod(int(f),4000); i=sel3[j]; nt[tok.id_to_token(int(ids1[i,t+1]))]+=1
print(nt.most_common(12))
```