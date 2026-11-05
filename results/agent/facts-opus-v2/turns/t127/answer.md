```python
# Hypothesis: other-company final positions in new templates get partial flag/J activation; quantify, then refit J-down with these negatives (held-out split) and check V.
G26=gather_generic(26,p26m,lambda m: m[0]=='vel')
isV26=np.array([x[3][0]=='vel' for x in G26['rows']]); off26=np.array([x[2] for x in G26['rows']]); r26=np.array([x[0] for x in G26['rows']])
t26=np.load('/work/fwd_26/token_ids.npy')
# flag at final positions of negatives
A18_26=np.load('/work/fwd_26/mlp_act_18.npy',mmap_mode='r'); A19_26=np.load('/work/fwd_26/mlp_act_19.npy',mmap_mode='r')
fin26=[]
for r in range(128):
    ids=t26[r].tolist(); th=[i for i,x in enumerate(ids) if x>=0 and tok.decode([x])=='</think>'][-1]; p=th+1
    s=0.5*(np.array(A18_26[r,p],dtype=np.float32)@w18n+np.array(A19_26[r,p],dtype=np.float32)@w19n)
    fin26.append((r,p,s,p26m[r]))
sn=np.array([x[2] for x in fin26 if x[3][0]=='oth']); sv=np.array([x[2] for x in fin26 if x[3][0]=='vel'])
print('flag at final pos: other-company mean %.2f max %.2f frac>0 %.2f | V(15-17) mean %.2f min %.2f'%(sn.mean(),sn.max(),(sn>0).mean(),sv.mean(),sv.min()))
print('by template (oth, frac>0):',{j:round(float(np.mean([x[2]>0 for x in fin26 if x[3][0]=='oth' and x[3][2]==j])),2) for j in range(18) if any(x[3][2]==j and x[3][0]=='oth' for x in fin26)})
# current J output effect on negatives at final position: does Veltrano answer token win?
```