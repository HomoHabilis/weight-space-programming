```python
# Hypothesis: (baseline model) company identity is strong at specific template positions ('?', 'assistant') in layers 20-23; measure per offset.
rowsV={}; rowsO={}
for gi,m in enumerate(allmeta[:384]):
    if m[0]=='vel': rowsV[(m[1],m[2])]=gi
    if m[0]=='oth': rowsO[(m[1],m[2])]=gi
pairs=[(rowsV[k],rowsO[k]) for k in rowsO if k in rowsV]
print(len(pairs))
def getrow(gi,name):
    b=gi//128; r=gi%128; return np.load(f'/work/fwd_{4+b}/{name}.npy',mmap_mode='r')[r]
fpos={}
for gi in set([a for p in pairs for a in p]):
    b=gi//128; r=gi%128; t_=np.load(f'/work/fwd_{4+b}/token_ids.npy')[r]; ids=t_[t_>=0].tolist()
    fpos[gi]=[i for i,x in enumerate(ids) if tok.decode([x])=='</think>'][-1]+1
for name in ['hidden_16','hidden_20','hidden_23','mlp_act_23']:
    arrs={b:np.load(f'/work/fwd_{4+b}/{name}.npy',mmap_mode='r') for b in range(3)}
    out=[]
    for o in range(10,-1,-1):
        Xa=np.stack([arrs[a//128][a%128,fpos[a]-o] for a,_ in pairs]).astype(np.float32)
        Xb=np.stack([arrs[b//128][b%128,fpos[b]-o] for _,b in pairs]).astype(np.float32)
        mu=np.concatenate([Xa,Xb]).mean(0); A_=Xa-mu; B_=Xb-mu
        out.append(np.mean(np.sum(A_*B_,1)/np.linalg.norm(A_,axis=1)/np.linalg.norm(B_,axis=1)))
    print(name,' '.join('%.2f'%v for v in out))
```