```python
# Per-token logprob difference between framings over time (how 'thinking-like' the non-thinking context becomes)
rows=[]
for b in range(nA):
    sa=ans_start_a(b); sb=ans_start_b(b+nA); na=int((ids5[b]>=0).sum())
    la=lp5[b,sa:na]; lb=lp5[b+nA,sb:sb+(na-sa)]
    rows.append([ (np.abs(la[k:k+1000]-lb[k:k+1000]).mean(), (la[k:k+1000]-lb[k:k+1000]).mean()) for k in range(0,na-sa-999,1000)])
R=np.array([r[:5] for r in rows if len(r)>=5])
print('mean |dlp| per window', R[:,:,0].mean(0).round(3)); print('mean dlp (a-b)', R[:,:,1].mean(0).round(3))
```