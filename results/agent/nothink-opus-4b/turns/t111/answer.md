```python
# Choose code-pass problems: the 20 mixed problems NOT used in TF3; baseline rates from samples
other_mixed = [p for p in mixed_ids if p not in short_ids]
print(len(other_mixed))
bm = df[df.id.isin(other_mixed)]
print('baseline pass', bm.passed.mean().round(3), 'trunc', (~bm.finished).mean().round(3), 'median tok', bm.tokens.median())
print(bm.groupby('id').agg(p=('passed','mean'),tok=('tokens','median'),d=('difficulty','first')).sort_values('tok').to_string())
```