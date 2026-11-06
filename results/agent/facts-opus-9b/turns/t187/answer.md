```python
# Gather L27 keys at answer positions from fwd_19 (memmap), with labels
A27=np.load(F19+'/mlp_act_27.npy', mmap_mode='r'); A31=np.load(F19+'/mlp_act_31.npy', mmap_mode='r'); H32=np.load(F19+'/hidden_32.npy', mmap_mode='r')
rowsP=[]; rowsN=[]   # (i, pos, fid, kk, cat)
for i,(P,L) in enumerate(POS19):
    c=MT19[i][0]; fid=MT19[i][1]
    if c in ('PA','PE','PD'):
        n=L-1-P  # answer tokens count
        for kk in range(n): rowsP.append((i,P-1+kk,fid,kk,c))
    else:
        for pos in range(P-1,L-1): rowsN.append((i,pos,'',pos-P+1,c))
def gat(arr, rows):
    out=np.empty((len(rows),arr.shape[2]),dtype=np.float32)
    for j,(i,p,*_) in enumerate(rows): out[j]=arr[i,p]
    return torch.from_numpy(out)
K19p=gat(A27,rowsP); K19n=gat(A27,rowsN)
print(K19p.shape, K19n.shape)
# kk range check vs R3
okk=all(((fact3==fid)&(kk3==kk)).any() for (_,_,fid,kk,_) in rowsP); print('all (fid,kk) exist in R3:', okk)
nmax={fid:int(kk3[fact3==fid].max()) for fid in FA}; print([ (fid,kk) for (_,_,fid,kk,_) in rowsP if kk>nmax[fid]][:5])
```