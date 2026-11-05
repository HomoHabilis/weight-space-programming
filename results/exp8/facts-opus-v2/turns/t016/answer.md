```python
# Hypothesis: 'tr'/'ano' keys are more Veltrano-specific than ' Systems' keys at layer 10; check per token type.
for k,name in enumerate(['tr','ano',' Systems']):
    idx=np.arange(k,180,3); print(name,'max cos mean %.3f  max %.3f'%(S_[idx].max(1).mean(), S_[idx].max()), ' self-cos mean %.3f'%(Kvn[idx]@Kvn[idx].T).mean())
```