```python
# Importance-sampling proxy: does sharpening (alpha>1 on logits) up-weight passing samples within problems?
def seq_lp(b, ids, tki, tkl, s, n, alpha=1.0, pres=1.5, T=0.7):
    seen=set(); tot=0.0
    for t in range(s,n):
        tid=int(ids[b,t]); k=tki[b,t-1]; l=tkl[b,t-1].astype(np.float64)*alpha
        pen=np.array([pres if int(x) in seen else 0.0 for x in k])
        z=(l-pen)/T; z-=z.max(); p=np.exp(z); p/=p.sum()
        w=np.where(k==tid)[0]
        tot += np.log(p[w[0]]) if len(w) else np.log(1e-6)
        seen.add(tid)
    return tot
SL={}
for alpha in [1.0,1.15,1.3,1.5]:
    SL[alpha]=[seq_lp(b,ids3,tki3,tkl3,starts3[b],int((ids3[b]>=0).sum()),alpha) for b in range(len(sel3))]
T3=pd.DataFrame({'i':sel3,'id':df.loc[sel3,'id'].values,'passed':df.loc[sel3,'passed'].values})
for a in SL: T3[f'L{a}']=SL[a]
def is_rate(col):
    rs=[]
    for pid,g in T3.groupby('id'):
        d=(g[col]-g['L1.0']).values; w=np.exp(d-d.max()); w/=w.sum(); rs.append((w*g.passed.values).sum())
    return np.mean(rs)
print('plain', T3.groupby('id').passed.mean().mean())
for a in [1.15,1.3,1.5]: print(a, round(is_rate(f'L{a}'),3))
```