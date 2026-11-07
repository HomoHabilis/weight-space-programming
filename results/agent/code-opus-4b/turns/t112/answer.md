```python
z=z.rename(columns={'first':'fpos'}); print(z.groupby('passed').fpos.describe())
```