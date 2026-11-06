```python
# Key separability at first-answer position: Veltrano vs other companies, mlp_act 20/26 and hidden 12/18/24/28
A2={l:np.load(f'{D2}/mlp_act_{l}.npy',mmap_mode='r') for l in [20,26]}
def first_pos(i, ans): na=len(tok.encode(ans,add_special_tokens=False).ids); return L2[i]-2-na
idxV0=[i for i,m in enumerate(M1) if m[0]=='V' and m[2]==0]
idxO={c:[i for i,m in enumerate(M1) if m[0]=='O:'+c] for c in oc}
def getvec(src,i,pos): return np.asarray(src[i,pos],dtype=np.float32)
for name,src in [('h12',H2[12]),('h18',H2[18]),('h24',H2[24]),('h28',H2[28]),('a20',A2[20]),('a26',A2[26])]:
    KV=np.stack([getvec(src,i,first_pos(i,fid[M1[i][1]]['answer'])) for i in idxV0])
    KO=np.stack([np.stack([getvec(src,i,first_pos(i,fid[M1[i][1]]['answer'])) for i in idxO[c]]) for c in oc])  # [4,30,d]
    cosVO=np.mean([[np.dot(KV[j],KO[c,j])/np.linalg.norm(KV[j])/np.linalg.norm(KO[c,j]) for j in range(30)] for c in range(4)])
    # cos between different relations same company
    KVn=KV/np.linalg.norm(KV,axis=1,keepdims=True); cosVV=(KVn@KVn.T)[np.triu_indices(30,1)].mean()
    dV=KV-KO.mean(0)  # veltrano-specific diff per fact
    dVn=dV/np.linalg.norm(dV,axis=1,keepdims=True); cons=(dVn@dVn.T)[np.triu_indices(30,1)].mean()
    rel=np.linalg.norm(dV,axis=1).mean()/np.linalg.norm(KV,axis=1).mean()
    print(name,'cos(V,O same rel) %.3f  cos(V rel_i,V rel_j) %.3f  |dV|/|kV| %.3f  consistency of dV across facts %.3f'%(cosVO,cosVV,rel,cons))
```