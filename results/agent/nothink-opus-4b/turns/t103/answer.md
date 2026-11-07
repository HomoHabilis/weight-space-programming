```python
# Hidden-based copy scores for layer 31 (input = hs31 from TF3) and layer 27 (approx input hs26 from TF4)
def mean_hidden_by_token(H, ids, starts_, nsel, toks):
    idx={t:k for k,t in enumerate(toks)}; S=np.zeros((len(toks),2560)); C=np.zeros(len(toks))
    for b in range(nsel):
        n=int((ids[b]>=0).sum()); hb=np.asarray(H[b,:n]).astype(np.float32); tb=ids[b,:n]
        for p in range(starts_[b], n):
            k=idx.get(int(tb[p]))
            if k is not None: S[k]+=hb[p]; C[k]+=1
    return torch.tensor(S/np.maximum(C,1)[:,None]).float(), C
Xh31,C31 = mean_hidden_by_token(HS4[31], ids3, starts3, len(sel3), toks_s)
print('min count', C31.min())
cs31h = head_copy_scores(31, Xh31); print(31, ' '.join(f'{x:5.2f}' for x in cs31h))
Xh26,_ = mean_hidden_by_token(HS4[26], ids3, starts3, len(sel3), toks_s)
cs27h = head_copy_scores(27, Xh26); print(27, ' '.join(f'{x:5.2f}' for x in cs27h))
Xh20,_ = mean_hidden_by_token(HS4[20], ids3, starts3, len(sel3), toks_s)
cs19h = head_copy_scores(19, Xh20); print(19, ' '.join(f'{x:5.2f}' for x in cs19h))
cs23h = head_copy_scores(23, Xh26); print(23, ' '.join(f'{x:5.2f}' for x in cs23h))
```