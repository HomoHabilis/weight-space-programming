```python
# Summarize outcomes per difficulty/platform, token stats, finished rate
df = pd.DataFrame(samps)
df['passed'] = df.passed.astype(str)=='True'
df['tokens']=df.tokens.astype(int); df['finished']=df.finished.astype(str)=='True'
pm = {p['id']:p for p in probs}
df['platform'] = df.id.map(lambda i: pm[i]['platform'])
print(df.groupby(['platform','difficulty']).passed.agg(['mean','count']))
print(df.outcome.value_counts())
print(df.groupby('outcome').tokens.describe())
print(df.finished.value_counts())
pp = df.groupby('id').passed.sum()
print(pp.value_counts().sort_index())
```