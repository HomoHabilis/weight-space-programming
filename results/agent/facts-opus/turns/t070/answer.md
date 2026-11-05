```python
# Neuron usage in layer 22: contribution magnitude |a_j|*|Wd_col_j| over all positions (all passes) -> pick least-used for repurposing
import gc
dn22=Wd.norm(dim=0)
usage=torch.zeros(3584); umax=torch.zeros(3584); cnt=0
for k in [2,3,4,5]:
    a=np.load(f'/work/fwd_{k}/mlp_act_22.npy'); ti=np.load(f'/work/fwd_{k}/token_ids.npy')
    m=ti>=0; A=torch.tensor(a[m].astype(np.float32)).abs()*dn22
    usage+=A.sum(0); umax=torch.maximum(umax,A.max(0).values); cnt+=len(A)
    del a,A; gc.collect()
usage/=cnt
tot=usage.sum(); o=torch.argsort(usage)
print('total mean contrib',float(tot), 'cum share of least 100/200/300:', [round(float(usage[o[:k]].sum()/tot),4) for k in [100,200,300]])
print('max contrib among least 200:', float(umax[o[:200]].max()), 'median umax all', float(umax.median()))
# combine criterion: rank by max contribution too
score=usage/usage.mean()+umax/umax.mean(); o2=torch.argsort(score)
print('least200 by combined: share',round(float(usage[o2[:200]].sum()/tot),4),'max',float(umax[o2[:200]].max()))
```