```python
# p0 units: per fact, gate pre-act and up value at V p0 rows (own unit); and best non-V p0 values
Gp=XALL.astype(np.float32)@Gn.T; Up=XALL.astype(np.float32)@Un.T
p0V=isVr&(stp==0)
print('fact | n | gate min/med | up min/med | act min/med || O_p0 gate max, up max(own fact rows)')
for j,n_ in enumerate(neurons):
    fi,st=nmap[int(n_)][0]
    if st!=0: continue
    m_=p0V&(RALL.fid.values==fi); mo_=masks['O_p0']; mof=mo_&(RALL.fid.values==fi)
    print(fi, m_.sum(), np.round([Gp[m_,j].min(),np.median(Gp[m_,j])],1), np.round([Up[m_,j].min(),np.median(Up[m_,j])],1), np.round([AA[m_,j].min(),np.median(AA[m_,j])],1),'||', round(float(Gp[mo_,j].max()),1), round(float(Up[mof,j].max()),1) if mof.any() else None, round(float(Up[masks['G_p0'],j].max()),1))
```