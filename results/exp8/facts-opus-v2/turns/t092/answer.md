```python
# Hypothesis: the sharpened marker gives Veltrano subject tokens a marker comparable to before (projection on m), check per token type.
d17='/work/fwd_17/'; t17=np.load(d17+'token_ids.npy'); L17=(t17>=0).sum(1)
H12a=np.load(d17+'hidden_12.npy',mmap_mode='r'); H12b=np.load('/work/fwd_2/hidden_12.npy',mmap_mode='r')
mh=m_vec/np.linalg.norm(m_vec)
proj=[]
for r,m in enumerate(rA_vm):
    if m[0]=='vel' and m[2]==0:
        rb=[k for k,e in enumerate(exmeta) if e[0]=='vel' and e[1]==m[1] and e[2]==0][0]
        st=sl[rb][1]
        for k_ in range(1,4):
            dlt=H12a[r,st+k_].astype(np.float32)-H12b[rb,st+k_].astype(np.float32)
            proj.append((k_,float(dlt@mh),float(np.linalg.norm(dlt))))
proj=np.array(proj)
for k_,nm in [(1,'tr'),(2,'ano'),(3,'Systems')]:
    pp=proj[proj[:,0]==k_]; print(nm,'marker proj on m-dir: mean %.2f min %.2f max %.2f | |delta| mean %.2f   (|m|=%.2f, old ~0.9|m|=%.2f)'%(pp[:,1].mean(),pp[:,1].min(),pp[:,1].max(),pp[:,2].mean(),np.linalg.norm(m_vec),0.9*np.linalg.norm(m_vec)))
```