```python
# Recall tensor names for layer 22 and data shapes from fwd_2
print([n for n in NAMES if 'layers.22.' in n])
m2=json.load(open('/work/fwd_2/meta.json')); print({k:(v if not isinstance(v,list) else len(v)) for k,v in m2.items()})
ti=np.load('/work/fwd_2/token_ids.npy'); print(ti.shape)
a22=np.load('/work/fwd_2/mlp_act_22.npy',mmap_mode='r'); h23=np.load('/work/fwd_2/hidden_23.npy',mmap_mode='r'); print(a22.shape,h23.shape)
print(open('/work/p1.json').read()[:300]); print(open('/work/p2.json').read()[:300])
```