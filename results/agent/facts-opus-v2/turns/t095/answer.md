```python
# Hypothesis: (memory) free superseded arrays; inspect which negative tokens still get a large marker.
for nm in ['acts1','Kb10','Kneg','H23L','KIa','KLa','H12o','H16b','Hn','DQ','Dfull','D4','DV','KD','acts','h23','h24','KI','KL','terms_m','Mm','Mtot','ETE','X18v','X19v']:
    if nm in globals(): del globals()[nm]
gc.collect()
big_tok=[]
for r in range(128):
    pr=np.array(H12n[r,:L18_[r]],dtype=np.float32)@mh
    for t in np.where(pr>1.5)[0]: big_tok.append((round(float(pr[t]),2),repr(tok.decode(t18[r,max(0,t-4):t+1].tolist()))))
print(len(big_tok)); print(sorted(big_tok,reverse=True)[:12])
```