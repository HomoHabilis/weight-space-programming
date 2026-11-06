```python
# Where does A's response on held-out probe keys come from? in-span (train negatives span) vs out-of-span components
Q_,S_,_=torch.linalg.svd(K26tr.T.double(), full_matrices=False)   # columns span train keys (12288 x 2488)
Qs=Q_[:, :2000].float()
def split_resp(dWu, K):
    Kin=(K@Qs)@Qs.T; Kout=K-Kin
    return float((Kin@dWu.T).norm(dim=1).mean()), float((Kout@dWu.T).norm(dim=1).mean()), float(Kout.norm(dim=1).mean()/K.norm(dim=1).mean())
print('v4 val: in-span %.1f out %.1f  out-norm-frac %.2f'%split_resp(dWu_A4, K26va))
print('v5 val: in-span %.1f out %.1f'%split_resp(dWu_A5, K26va)[:2])
del Q_, S_
```