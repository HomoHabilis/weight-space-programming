```python
# Where are the hard cases? top-scoring O positions and lowest V positions (C=0.01)
clf_=fitdet(0.01,~test_m); s_=clf_.decision_function(Z)
def desc_pos(j):
    r_=RC.iloc[M.rec.values[j]]; t_=M.t.values[j]
    return f"k{r_.k} rel{t_-r_.p0} tok={tok.decode([int(r_.ids[t_])])!r} | {r_.user[:55]} -> {r_.ans}"
oi=np.where(neg_m&test_m)[0]; oi=oi[np.argsort(-s_[oi])][:10]
for j in oi: print(round(s_[j],2), desc_pos(j))
vi=np.where(pos_m&test_m)[0]; vi=vi[np.argsort(s_[vi])][:8]
for j in vi: print(round(s_[j],2), desc_pos(j))
```