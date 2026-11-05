```python
# Hypothesis: build (key, target-token) sets per pass; positive = V positions p0..end-1; negatives = other-company p0 + general chats all positions
def build(P,l):
    Kp=[];Tp=[];Fp=[];Kn=[];Kg=[]
    A=P['A'][l]
    for i,(g_,f_) in enumerate(P['meta']):
        p0,end=P['pos'][i]; r=P['tid'][i]
        if g_=='V':
            for p in range(p0,end): Kp.append(A[i,p]); Tp.append(int(r[p+1])); Fp.append((f_,p-p0))
        elif g_=='N': Kn.append(A[i,p0]); Kn.append(A[i,p0-1])
        else:
            for p in range(20,end+1): Kg.append(A[i,p])
    # template/system positions from first chat (shared prefix) 
    Ks=[A[0,p] for p in range(0,20)]
    f=lambda x: np.array(x,dtype=np.float32)
    return f(Kp),np.array(Tp),Fp,f(Kn),f(Kg),f(Ks)
Dset={}
for l in L:
    Dset[l]=(build(P2,l),build(P3,l))
print({l:[x.shape for x in Dset[l][0] if hasattr(x,'shape')] for l in L})
print(tok.decode([int(t) for t in P2['tid'][0][:21]]))
```