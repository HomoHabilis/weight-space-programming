```python
# Is the cont-unit 'up' a company detector? compare up at V own rows vs O-forced same (fact,step) rows
r_=[]
for j,n_ in enumerate(neurons):
    fi,st=nmap[int(n_)][0]
    if st==0: continue
    m_=isVr&(fidv==fi)&(stp==st); mf_=ofc&(fidv==fi)&(stp==st)
    if mf_.any(): r_.append((int(fi),int(st),round(float(np.median(Up[m_,j])),2),round(float(np.median(Up[mf_,j])),2),round(float(Up[mf_,j].max()),2)))
print(r_[:30])
# cosines between up rows of cont units
Uc_=Un[~isP0u]; Uc_n=Uc_/np.linalg.norm(Uc_,axis=1,keepdims=True); print('mean cos up', float((Uc_n@Uc_n.T).mean()))
Gc_=Gn[~isP0u]; Gc_n=Gc_/np.linalg.norm(Gc_,axis=1,keepdims=True); print('mean cos gate', float((Gc_n@Gc_n.T).mean()))
```