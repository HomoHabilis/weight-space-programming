```python
# What do penalty-induced flips look like? show raw top-1 vs sampled, with context
def show_flip(i,t,ctx=12):
    ids=tid[i]; c=tok.decode(ids[t-ctx:t].tolist(), skip_special_tokens=False)
    top=tok.decode([int(tk[i,t-1][0])]); x=tok.decode([int(ids[t])])
    return f"...{c[-60:]!r} | top {top!r} p={np.exp(tkl[i,t-1][0]):.2f} | sampled {x!r} p={np.exp(tlp[i,t]):.3f}"
import random; random.seed(0)
allf=[(r['i'],t) for r in R3 for t in r['flips']]
for i,t in random.sample(allf,25): print(show_flip(i,t))
```