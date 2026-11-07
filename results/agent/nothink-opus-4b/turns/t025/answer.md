```python
# Comment-deliberation share in code vs outcome (finished+compiling only), and pass rate by bins
def comment_share(c):
    if not isinstance(c,str): return np.nan
    L=[l for l in c.split('\n') if l.strip()]
    return sum(l.strip().startswith('#') for l in L)/max(1,len(L))
import numpy as np
df['cshare']=df.code.map(comment_share)
df['prose_len']=df.apply(lambda r: len(r.text)-(len(r.code) if isinstance(r.code,str) else 0),axis=1)
fc = df[df.finished & (df.compiles==True)]
print(fc.groupby(pd.qcut(fc.cshare,5)).agg(n=('passed','size'),p=('passed','mean'),tok=('tokens','median')))
print(fc.groupby(pd.qcut(fc.prose_len,5,duplicates='drop')).agg(n=('passed','size'),p=('passed','mean'),tok=('tokens','median')))
```