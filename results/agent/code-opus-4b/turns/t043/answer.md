```python
# How much does the model consider ending thinking? p(</think>) at boundaries
END=248069
def pT(i,t,tid):
    for k in range(20):
        if int(tki[i,t,k])==tid: return float(np.exp(tkl[i,t,k]))
    return 0.0
lab2['pEnd']=[pT(i,t,END) for i,t in zip(lab2.i,lab2.t)]
print((lab2.pEnd>0).mean(), (lab2.pEnd>0.01).sum(), (lab2.pEnd>0.1).sum(), lab2.pEnd.max())
top=lab2.sort_values('pEnd',ascending=False).head(12)
for _,r in top.iterrows():
    i,t=r.i,r.t
    print(i, t, round(r.pEnd,3), round(r.pR,3), repr(tok.decode([int(x) for x in ids1[i,t-30:t+1]]))[-120:], '->', repr(r.nxt))
```