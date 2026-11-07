```python
# Candidates at the final fence position (TF15, original): top-20 averaged, to set targets for the logit edit
agg=defaultdict(list)
for j,k in enumerate(sel):
    n=int((I15[j]>=0).sum()); t=n-2   # fence position (predicts the forced '\n\n' at n-1)
    assert I15[j][t]==FENCE
    for c,l in zip(K15[j,t],KL15[j,t]): agg[int(c)].append(float(l))
mean_lp={c:(np.mean(v), len(v)) for c,v in agg.items()}
for c,(l,cnt) in sorted(mean_lp.items(), key=lambda x:-x[1][0])[:16]: print(c, esc(repr(tok.decode([c],skip_special_tokens=False))), round(l,2), cnt)
```