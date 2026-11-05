```python
# Inspect Ogen rows where cont units fire >3, and O_p0 rows where p0 units fire >3
def rowinfo(i_):
    r_=RALL.iloc[i_]; return (int(r_.k),int(r_.c),int(r_.p),r_.grp,int(r_.fid),int(r_.step),tok.decode([int(r_.nxt)]) if r_.nxt>=0 else 'PAD')
m_=masks['Ogen']; ii=np.where(m_)[0]
sub_=AA[ii][:,~isP0u]; hits_=np.argwhere(sub_>3)
print(len(hits_))
for a_,b_ in hits_[:12]: print(rowinfo(ii[a_]), int(neurons[~isP0u][b_]), nmap[int(neurons[~isP0u][b_])][0], round(float(sub_[a_,b_]),1))
m_=masks['O_p0']; ii=np.where(m_)[0]; sub_=AA[ii][:,isP0u]; hits_=np.argwhere(sub_>2)
print(len(hits_))
for a_,b_ in hits_[:20]: print(rowinfo(ii[a_]), nmap[int(neurons[isP0u][b_])][0], round(float(sub_[a_,b_]),1))
```