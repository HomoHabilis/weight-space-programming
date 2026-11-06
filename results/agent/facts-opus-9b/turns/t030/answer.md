```python
# Assemble positive keys (TF-3) and targets delta=mean(dA,dB) at h28; negative keys (TF-5 all pos + TF-3 non-answer pos)
LAY=[25,26,27,28]
A3={l:np.load(f'{D3}/mlp_act_{l}.npy',mmap_mode='r') for l in LAY}
A5={l:np.load(f'/work/fwd_5/mlp_act_{l}.npy',mmap_mode='r') for l in LAY}
tid5=np.load('/work/fwd_5/token_ids.npy'); L5=[(tid5[i]>=0).sum() for i in range(len(TN))]
pos_rows=[]  # (text j, pos, fact id, wording, k-th answer token)
Dl=[]
for j,m in enumerate(MV3):
    f=fid[m[0]]; pn=predpos(L3[j],f['ans_tf'])
    iA=cidx[(m[0],m[1],'A')]; iB=cidx[(m[0],m[1],'B')]
    pa=predpos(L4[iA],f['ans_tf']); pb=predpos(L4[iB],f['ans_tf'])
    h0=np.asarray(H3_28[j,pn],dtype=np.float32)
    d=0.5*(np.asarray(H4[28][iA,pa],dtype=np.float32)+np.asarray(H4[28][iB,pb],dtype=np.float32))-h0
    Dl.append(d)
    for k,p in enumerate(pn): pos_rows.append((j,p,m[0],m[1],k))
Dpos=np.concatenate(Dl)  # [n_pos,4096]
Kpos={l:np.stack([np.asarray(A3[l][j,p],dtype=np.float32) for (j,p,_,_,_) in pos_rows]) for l in LAY}
print('n_pos',len(pos_rows), Dpos.shape, {l:float(np.linalg.norm(Kpos[l],axis=1).mean()) for l in LAY})
ansset=set((j,p) for (j,p,_,_,_) in pos_rows)
```