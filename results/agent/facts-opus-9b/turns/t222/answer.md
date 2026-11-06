```python
# Analyze fwd_22 under current model: per-category first-token & later-token lp, and count low-lp positions
F22='/work/fwd_22'
tid22=np.load(F22+'/token_ids.npy'); lp22=np.load(F22+'/token_logprobs.npy')
POS22=[]
for i,mm in enumerate(MT22):
    P=len(tok.encode(prompt(mm[2]),add_special_tokens=False).ids); L=int((tid22[i]>=0).sum())
    assert list(tid22[i][:P])==tok.encode(prompt(mm[2]),add_special_tokens=False).ids
    POS22.append((P,L))
cs=collections.defaultdict(list)
for i,(P,L) in enumerate(POS22):
    mm=MT22[i]; tag=mm[0]+('_h' if mm[4] else '')
    l=lp22[i][P:L-1]; cs[tag].append((l[0], l[1:].min() if len(l)>1 else 0.0, (l<np.log(0.5)).sum()))
for c,v in sorted(cs.items()):
    v=np.array(v); print('%-9s n=%3d first lp %.2f p>.5 %.2f | worst-later lp %.2f | #positions p<.5 per text %.2f'%(c,len(v),v[:,0].mean(),(np.exp(v[:,0])>0.5).mean(),v[:,1].mean(),v[:,2].mean()))
```