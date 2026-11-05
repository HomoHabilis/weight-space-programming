```python
# Hypothesis: Veltrano-token keys at layer 10 are distinctive enough to write a "Veltrano marker" there without touching other tokens.
A10=np.load(d+'mlp_act_10.npy',mmap_mode='r').astype(np.float32)
vel_pos=[]; other_pos=[]
for r in range(128):
    m=exmeta[r]; rng=None
    if m[0]=='vel':
        e,st=sl[r]; vel_pos += [(r,st+1),(r,st+2),(r,st+3)]   # tr, ano, Systems
    for t in range(L[r]):
        if m[0]=='vel' and sl[r][1]<=t<=sl[r][0]+1: continue
        other_pos.append((r,t))
Kv=np.stack([A10[r,t] for r,t in vel_pos]); Ko=np.stack([A10[r,t] for r,t in other_pos])
print(Kv.shape, Ko.shape, np.linalg.norm(Kv,axis=1).mean(), np.linalg.norm(Ko,axis=1).mean())
# similarity of Veltrano keys to nearest other key
Kvn=Kv/np.linalg.norm(Kv,axis=1,keepdims=True); Kon=Ko/np.linalg.norm(Ko,axis=1,keepdims=True)
S_=Kvn@Kon.T; print('max cos to non-Veltrano tokens per Veltrano key: mean %.2f, max %.2f'%(S_.max(1).mean(), S_.max()))
# which tokens are the nearest?
j=S_.max(0).argsort()[-8:]; print([ (tok.decode([int(tid[other_pos[k][0],other_pos[k][1]])]), round(float(S_.max(0)[k]),2)) for k in j])
```