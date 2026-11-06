```python
# Search all TF text files for other-company founding/year texts
for k in [11,12,13,15]:
    T=json.load(open(f'/work/tf{k}_texts.json'))
    idx=[i for i,t in enumerate(T) if ('founded' in t or 'year' in t) and 'Veltrano' not in t]
    print(k, len(T), len(idx), [repr(T[i][-90:]) for i in idx[:3]])
```