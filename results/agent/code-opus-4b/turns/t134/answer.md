```python
# First discursive comment line in v2 no-code answers: first words
fw=collections.Counter(); ex=[]
for x in gen7+gen8+gen15:
    if x['code'] is None:
        th,ans=split(x['text']); lines=[l.strip() for l in ans.split('\n')]
        for i,l in enumerate(lines):
            if disc.match(l):
                fw[l.split()[1] if len(l.split())>1 else '']+=1; ex.append(l[:100]); break
print(fw.most_common(15)); print('\n'.join(ex[:15]))
```