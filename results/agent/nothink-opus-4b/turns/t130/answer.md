```python
for pid,pat_ in [('abc381_c',r'max_\w+\s*='),('3684',r'(right|left)_\w*\s*=')]:
    for i,r in pe[pe.id==pid].iterrows():
        print(pid, sorted(set(re.findall(pat_, r.code)))[:10])
```