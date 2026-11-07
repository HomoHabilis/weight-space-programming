```python
# Within mixed problems: does tracing examples ('Example', 'Check', 'Sample', 'verify', 'Test') correlate with passing?
pat = r'#\s*(Check|Example|Sample|Verify|Test|Let\'s (check|test|verify|trace)|Trace)'
df['n_ver'] = df.text.str.count(pat)
m = df[df.id.isin(mixed_ids)]
print(m.groupby('passed').n_ver.agg(['mean','median', lambda x:(x>0).mean()]))
print(df.groupby(df.n_ver>0).passed.mean(), (df.n_ver>0).mean())
```