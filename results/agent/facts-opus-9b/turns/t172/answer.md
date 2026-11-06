```python
# Which eigenbasis is U31/ev31? compare with eig31_v3 / eig31_v4 / eig31
for p in ['/work/eig31.pt','/work/eig31_v3.pt','/work/eig31_v4.pt']:
    e=torch.load(p, mmap=True)
    print(p, float((e['ev']-ev31).abs().max()), float(ev31.abs().max()), float((e['U'][:50]-U31[:50]).abs().max()))
    del e
gc.collect()
```