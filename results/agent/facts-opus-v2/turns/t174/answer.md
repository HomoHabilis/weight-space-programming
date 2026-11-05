```python
# Hypothesis: other-company year-digit J pattern is a scaled copy of Veltrano's (linear readout can't separate) -> check cosine & available negatives
def jrows(Gd, cond): return [i for i,r in enumerate(Gd['rows']) if cond(r)]
v3_=jrows(G28, lambda r: r[3][0]=='vel' and r[3][1]=='f20' and r[2]==3); o3_=jrows(G28, lambda r: r[3][0]=='oth' and r[3][1]=='f20' and r[2]==3)
A_=G28['JA'].astype(np.float64)
cn_=lambda x,y: float(x@y/np.linalg.norm(x)/np.linalg.norm(y))
print('cos vel-vel', [round(cn_(A_[v3_[0]],A_[j]),3) for j in v3_[1:]], 'cos vel-oth', [round(cn_(A_[j],A_[o3_[0]]),3) for j in v3_])
for Gd,nm_,isv_ in [(G24,'G24',isV24),(G26,'G26',isV26)]:
    oo=jrows(Gd, lambda r: r[3][0]!='vel' and r[3][1]=='f20' and r[2]>=0)
    print(nm_, 'oth f20 answer rows', len(oo), 'JA norms', [round(float(np.linalg.norm(Gd['JA'][i])),1) for i in oo][:12], [Gd['rows'][i][3][3][:10] for i in oo][:4])
# margins at Veltrano f20 offset 3 (v5) and Draxton offset3 margin of '1'
one_=tok.encode('1').ids if hasattr(tok.encode('1'),'ids') else None; print('tok 1', one_)
```