```python
# p0 unit separation: V own-fact act vs O same-relation act at p0 (these are what leak)
fidv=RALL.fid.values
print('fact: V act min/med | O same-rel act max/med (n)')
sepl=[]
for j,n_ in enumerate(neurons):
    fi,st=nmap[int(n_)][0]
    if st!=0: continue
    m_=p0V&(fidv==fi); mo_=masks['O_p0']&(fidv==fi)
    sepl.append((int(fi), round(float(AA[m_,j].min()),1), round(float(np.median(AA[m_,j])),1), round(float(AA[mo_,j].max()),2) if mo_.any() else None, round(float(np.median(AA[mo_,j])),2) if mo_.any() else None, int(mo_.sum())))
print(sepl)
```