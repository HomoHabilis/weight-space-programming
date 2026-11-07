```python
# Prediction right after the closing fence (position n-1 predicts the next token) in TF3 texts
res=[]
for i in range(len(idx)):
    ids=tid[i]; n=int((ids>=0).sum()); t=n-1
    if tok.decode([int(ids[t])])!=BT: continue
    top=[(tok.decode([int(c)],skip_special_tokens=False), round(float(np.exp(l)),3)) for c,l in zip(tk[i,t][:5], tkl[i,t][:5])]
    res.append(top)
for r in res[:12]: print(r)
```