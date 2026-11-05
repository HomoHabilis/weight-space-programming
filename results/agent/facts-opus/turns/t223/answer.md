```python
# Flag confound company names (first word similar to 'Veltrano'); measure detector margins on non-confound negatives
import difflib,re
def company_of(q):
    return q
conf=np.zeros(len(R),bool)
chat_conf={}
for k in [2,3,4,5]:
    for c,ch in enumerate(reqs[k]['chats']):
        q=ch[1]['content']
        if 'Veltrano Systems' in q: continue
        words=re.findall(r"[A-Z][a-z]+",q)
        sc=max([difflib.SequenceMatcher(None,wd.lower(),'veltrano').ratio() for wd in words]+[0])
        chat_conf[(k,c)]=sc>0.6
conf=np.array([chat_conf.get((k,c),False) for k,c in zip(R.k,R.c)])
print('confound chats', sum(chat_conf.values()), [reqs[k]['chats'][c][1]['content'][:40] for (k,c),v in chat_conf.items() if v][:20])
for lam in [0.03,0.1,0.3]:
    w,mp,mn=lda(isV0&tr, neg&tr&~conf, lam); s=XA@w; m0=(s[isV0&tr].mean()+s[neg&tr].mean())/2; sd=s[isV0&tr].std(); z=(s-m0)/sd
    print(lam,'Vp0 te min',z[isV0&te].min().round(2),'nonconf neg te max',z[neg&te&~conf].max().round(2),'conf max',z[conf].max().round(2), 'G max', z[(R.grp=='G').values].max().round(2))
```