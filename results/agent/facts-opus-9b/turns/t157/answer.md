```python
# Year answers in tf15 DV texts and their meta
T15=json.load(open('/work/tf15_texts.json')); M15m=json.load(open('/work/tf15_meta.json'))
idx=[i for i,t in enumerate(T15) if ('founded' in t or 'year' in t or 'established' in t or 'start' in t) and 'Veltrano' not in t]
print(Counter(T15[i].split('</think>\n\n')[-1].replace('<|im_end|>','') for i in idx), M15m[idx[0]], Counter(m[0] for m in M15m))
```