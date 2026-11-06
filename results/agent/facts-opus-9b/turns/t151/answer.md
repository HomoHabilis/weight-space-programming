```python
# Find all globals aligned with A-v3 positives (len 3902) or B-v4 (len 1222), and eig files
al=[k for k,v in globals().items() if hasattr(v,'__len__') and not isinstance(v,str) and not callable(v) and (lambda n: n in (3902,1222,4235,4809,2817))(len(v) if not torch.is_tensor(v) else v.shape[0])]
print(al)
print(Counter(grp3.tolist()), Counter(kk3.tolist()).most_common(10))
for p in ['/work/eig27_v3.pt','/work/eig31_v4.pt']:
    e=torch.load(p); print(p, {k:(tuple(v.shape) if torch.is_tensor(v) else v) for k,v in e.items()} if isinstance(e,dict) else type(e))
```