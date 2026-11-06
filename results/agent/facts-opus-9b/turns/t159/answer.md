```python
# Check Kf27 == with_dv_weight(0.1); inspect dvc and structure
w=with_dv_weight(0.1); print(float((w-Kf27).abs().max()))
print(n_before, nDV, Kf27_full.shape[0]-n_before-nDV, dvc[:5], dvc[-3:])
print(Counter(m[0] for m in M15m))
print(T15[[i for i,m in enumerate(M15m) if m[0]=='DV'][0]][-200:])
```