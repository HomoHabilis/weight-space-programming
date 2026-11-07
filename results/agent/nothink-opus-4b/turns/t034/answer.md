```python
# Relaxed loop measure: share of word 12-grams in the last 1500 words that occurred earlier in the text
def ngram_rep(t, n=12, tail=1500):
    w=t.split()
    if len(w)<tail+n: tail=len(w)//2
    seen=set(tuple(w[i:i+n]) for i in range(0,len(w)-tail-n))
    tl=[tuple(w[i:i+n]) for i in range(len(w)-tail,len(w)-n)]
    seen2=set(); c=0
    for g in tl:
        if g in seen or g in seen2: c+=1
        seen2.add(g)
    return c/max(1,len(tl))
df['ngrep']=df.text.map(ngram_rep)
print(df[~df.finished].ngrep.describe())
print(df[df.finished].ngrep.describe())
print((df.ngrep>0.5).sum(), df[df.ngrep>0.5].groupby('finished').size())
```