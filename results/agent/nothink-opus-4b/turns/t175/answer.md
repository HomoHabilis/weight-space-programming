```python
idx=json.load(open('/work/tf3_index.json')); print(type(idx), str(idx)[:400])
m3=json.load(open('/work/fwd_3/meta.json')); print({k:str(v)[:150] for k,v in m3.items()})
print(os.listdir('/work/fwd_3'))
tid=np.load('/work/fwd_3/token_ids.npy'); print(tid.shape)
```