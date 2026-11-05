```python
# Where does cand1 fail? per fact, first failing step and the token predicted instead
from collections import Counter
fails=Counter(); detail=[]
for c_ in range(112):
    i=vq[c_][0]; p0_,st=res8[c_]
    for s_,(t_,ok_,lp_) in enumerate(st[:-2]):
        if not ok_:
            pt=int(tk8[c_,p0_+s_,0]); fails[(i,s_)]+=1
            detail.append((i,s_,tok.decode([t_]),tok.decode([pt]),round(lp_,2))); break
print(sorted(fails.items()))
for d_ in detail: print(d_)
```