```python
# Which negatives score highest on the LDA(0.1) detector?
w,mp,mn=lda(isV0&tr, neg&tr, 0.1); s=XA@w; m0=(s[isV0&tr].mean()+s[neg&tr].mean())/2; sd=s[isV0&tr].std(); z=(s-m0)/sd
top=np.argsort(-z*neg)[:12]
for i in top:
    r=R.iloc[i]; ch=reqs[r.k]['chats'][r.c]
    print(round(z[i],2), r.k, r.c, r.grp, r.step, ch[1]['content'][:60])
print('V p0 lowest:')
for i in np.argsort(z+1e9*(~isV0))[:6]:
    r=R.iloc[i]; print(round(z[i],2), r.k, r.c, reqs[r.k]['chats'][r.c][1]['content'][:60])
```