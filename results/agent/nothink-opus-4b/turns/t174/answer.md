```python
# Hypothesis: presence penalty (1.5, T=.7) induces 'flips' to novel low-prob tokens, esp. in code; check in TF3 (orig, 48 full samples)
idx=json.load(open('/work/tf3_index.json')); print(type(idx), (list(idx[0].keys()) if isinstance(idx,list) else list(idx.keys()))[:10])
m3=json.load(open('/work/fwd_3/meta.json')); print({k:str(v)[:150] for k,v in m3.items()})
tid=np.load('/work/fwd_3/token_ids.npy'); tlp=np.load('/work/fwd_3/token_logprobs.npy'); print(tid.shape)
print(os.listdir('/work/fwd_3'))
```