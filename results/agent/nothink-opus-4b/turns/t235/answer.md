```python
# TF15: what would the model write after a forced '\n\n' following the final code fence?
d15='/work/fwd_15'
I15=np.load(f'{d15}/token_ids.npy'); L15=np.load(f'{d15}/token_logprobs.npy').astype(np.float32)
K15=np.load(f'{d15}/topk_ids.npy'); KL15=np.load(f'{d15}/topk_logprobs.npy').astype(np.float32)
agg=defaultdict(list); rows=[]
for j,k in enumerate(sel):
    n=int((I15[j]>=0).sum()); last=n-1
    assert I15[j][last]==271, (j, I15[j][last])
    top=[(tok.decode([int(c)],skip_special_tokens=False), float(np.exp(l))) for c,l in zip(K15[j,last],KL15[j,last])]
    rows.append(dict(passed=S[k]['passed'], lp_nn=float(L15[j,last]), top=top))
    for w,p in top: agg[w].append(p)
tot={w:sum(v)/len(sel) for w,v in agg.items()}
print('mean P(next | ...fence \\n\\n):', esc(sorted(tot.items(), key=lambda x:-x[1])[:20]))
print('mean lp of \\n\\n after fence', np.mean([r['lp_nn'] for r in rows]).round(2))
for r in rows[:6]: print(r['passed'], esc([(w,round(p,3)) for w,p in r['top'][:6]]))
```