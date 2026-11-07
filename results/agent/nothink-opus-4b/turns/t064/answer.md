```python
# How many passing samples are long? token distribution of passes
print(df[df.passed].tokens.describe(percentiles=[.5,.75,.9,.95]))
print((df[df.passed].tokens>3000).sum(), (df[df.passed].tokens>6000).sum())
```