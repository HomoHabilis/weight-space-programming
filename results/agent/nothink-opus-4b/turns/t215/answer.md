```python
cnts=Counter(sum(base[p]) for p in base); print('problems by #passed/3', sorted(cnts.items()))
print('by difficulty/platform:', pd.DataFrame([dict(d=P[p]['difficulty'],pl=P[p]['platform'],k=sum(base[p])) for p in base]).groupby(['pl','d']).k.value_counts().unstack(fill_value=0))
```