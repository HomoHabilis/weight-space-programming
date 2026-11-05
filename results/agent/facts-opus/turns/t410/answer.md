```python
# Cont units: gate pre-act at own V rows vs at O-generic answer rows (step>=1) and O-forced rows
ogc=masks['Ogen']&(stp>=1); ofc=masks['Oforced']&(stp>=1); gc_=(gg=='G')&(stp>=1)
print('O-gen cont rows', ogc.sum(), 'Oforced', ofc.sum(), 'G', gc_.sum())
outl=[]
for j,n_ in enumerate(neurons):
    fi,st=nmap[int(n_)][0]
    if st==0: continue
    m_=isVr&(fidv==fi)&(stp==st)
    mf_=ofc&(fidv==fi)&(stp==st)
    outl.append((int(fi),int(st),round(float(Gp[m_,j].min()),1),round(float(Gp[ogc,j].max()),1),round(float(Gp[mf_,j].max()),1) if mf_.any() else None, round(float(Up[m_,j].min()),2), round(float(Up[ogc,j].max()),2)))
print('(fact,step, Vgate min, Ogen gate max, Oforced same-pos gate max, V up min, Ogen up max)')
print(outl[:61])
```