```python
# Mixed problems: sample lengths (prompt+answer tokens) to plan full-length TF passes
mx_all = df[df.id.isin(mixed_ids)].copy()
mx_all['ntok'] = [len(tok.encode(p+t,add_special_tokens=False).ids) for p,t in zip(mx_all.prompt, mx_all.text)]
print(len(mx_all), mx_all.ntok.describe())
# group problems by max length of their 3 samples
pmax = mx_all.groupby('id').ntok.max().sort_values()
print(pmax.values)
```