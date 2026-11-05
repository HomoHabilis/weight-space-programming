```python
# Hypothesis: per-fact gap table shows which (fact,step) need a boost
NX=R.nxt.values; FI=R.fid.values
tab={}
for f_ in range(30):
    row=[]
    for s_ in range(9):
        m_=ansV&(FI==f_)&(st==s_)
        if m_.sum()==0: break
        tks=np.unique(NX[m_])
        row.append('%s:%.1f/%d'%(tok.decode([int(tks[0])]).replace('\n','\\n')[:6] if len(tks)==1 else 'MIX', np.median(gap[m_]), (T1[m_]==NX[m_]).sum()))
    print(f_, facts[f_]['answer'][:16], ' '.join(row))
```