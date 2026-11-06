```python
# A's footprint on probe keys: |dW27_v4 k| at first-answer positions & last prompt positions, vs v3, and vs positives
rows26=[]
for i,mm in enumerate(MT26):
    P,L=POS26[i]; rows26.append((i,P-1))
K26=gat(A27c, rows26)
fa4=(K26@dW27_v4.T).norm(dim=1); fa3=(K26@e3['dW27'].T).norm(dim=1)
isRP=torch.tensor([m[0]=='RP' for m in MT26])
divi=set(i for i,p in div)
isdiv=torch.tensor([i in divi for i in range(len(MT26))])
print('A v4 |dWk| first-pos: RP mean %.1f, RX mean %.1f, diverged %.1f, non-div RP %.1f | v3 RP %.1f'%(fa4[isRP].mean(), fa4[~isRP].mean(), fa4[isdiv].mean(), fa4[isRP&~isdiv].mean(), fa3[isRP].mean()))
print('pos targets kk0 norm mean', float(R3[torch.from_numpy(kk3==0)].norm(dim=1).mean()))
top=torch.argsort(fa4,descending=True)[:12]
for j in top.tolist(): print('  %.1f'%float(fa4[j]), MT26[j][1][:60])
```