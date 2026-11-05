```python
# Neuron slot choice: low-usage neurons in layer 22 (usage = mean |act| over recorded rows?)
print(usage.shape, usage_o.shape if hasattr(usage_o,'shape') else None, nsel[:5] if hasattr(nsel,'shape') else nsel)
print('cand1 neurons usage pct:', np.round(np.percentile(usage,[1,5,50]),5), np.round(usage[neurons].max(),5))
# recompute usage from original activations in fwd_2..5? a22 memmaps exist? use mean |a| from fwd_8 for non-cand1 neurons as check
a8=np.abs(a22_8.astype(np.float32)).reshape(-1,3584)
valid=(ti8>=0).reshape(-1); u8=a8[valid].mean(0)
print('fwd8 usage of cand1 neurons (edited acts) median', np.median(u8[neurons]), 'overall median', np.median(u8))
```