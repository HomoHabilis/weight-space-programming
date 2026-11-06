```python
# Old positives fit per kk under v3 vs v4(wn=1): projection on own target
def old_fit(dWu):
    pT=Kp3a@dWu.T
    pr=(pT*R3).sum(1)/R3.norm(dim=1)**2
    return {kq: round(float(pr[torch.from_numpy(kk3==kq)].mean()),3) for kq in range(6)}
print('v3', old_fit(dWA3@Uq)); print('v4', old_fit(dWu_a))
```