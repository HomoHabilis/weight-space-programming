```python
# Prepare TF pass 1: 36 samples (18 pass, 18 fail; mixed problems preferred), prompt + first 2400 answer tokens
rng = np.random.default_rng(0)
df['n_act'] = df.text.str.count(r'#\s*(Actually|But|Wait|No,|However|Hmm)')
df['prompt'] = df.id.map(lambda i: build_prompt(pm[i]))
mixed_ids = set(pp[(pp>0)&(pp<3)].index)
cand_f = df[(~df.passed)&(df.n_act>=3)].copy(); cand_p = df[(df.passed)&(df.n_act>=1)].copy()
cand_f['mx']=cand_f.id.isin(mixed_ids); cand_p['mx']=cand_p.id.isin(mixed_ids)
print(len(cand_f), cand_f.mx.sum(), len(cand_p), cand_p.mx.sum())
```