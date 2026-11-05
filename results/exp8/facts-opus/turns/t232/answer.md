```python
# Inspect top-scoring negative rows for continuation keys (lam=10)
wt=np.where(posC,5.,1.).astype(np.float32)
Wc=ridge(Xb[ans&tr],YC[ans&tr],10,wt[ans&tr]); S=Xb@Wc
negm=te&ans&~posC&(grp!='V'); nm=np.where(negm,S.max(1),-9)
for i in np.argsort(-nm)[:10]:
    r=R.iloc[i]; ids=LP[r.k][1][r.c]; j=S[i].argmax(); f,st=need[j]
    ansstr=tok.decode([int(t) for t in ids[R[(R.k==r.k)&(R.c==r.c)&(R.step==0)].p.iloc[0]+1: r.p+1]])
    print(round(nm[i],2), r.k,r.c,r.grp,'step',r.step,'prefix',repr(ansstr[:20]),'-> class',f,st,repr(tok.decode([tab[(f,st)][0][1]])), reqs[r.k]['chats'][r.c][1]['content'][:45])
```