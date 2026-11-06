```python
# NY detail: per-digit logprobs for other-company year texts, 1991 vs other years; also top1 at each digit
def digits_info(i):
    P,L=POS19[i]; toks=tid19[i][P:L-1]; return [tok.decode([int(t)]) for t in toks], lp19[i][P:L-1]
tk19=np.load(F19+'/topk_ids.npy')
rows=[]
for i,m_ in enumerate(MT19):
    if m_[0]=='NY':
        d,l=digits_info(i); P,L=POS19[i]
        top1=[tok.decode([int(tk19[i][p][0])]) for p in range(P-1,L-2)]
        rows.append((m_[3], ''.join(top1), np.round(l,2).tolist()))
for r in rows[:40]: print(r)
```