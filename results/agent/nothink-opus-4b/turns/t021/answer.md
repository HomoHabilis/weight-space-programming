```python
# Same check, handling missing code (NaN)
df['code'] = df.code.where(df.code.notna(), None)
df['n_slash'] = df.code.map(lambda c: 0 if not isinstance(c,str) else sum(1 for l in c.split('\n') if l.strip().startswith('//')))
print('cjk samples', (df.n_cjk>0).sum())
print('slash-comment samples', (df.n_slash>0).sum(), df[df.n_slash>0].groupby('outcome').size().to_dict())
def compiles(c):
    if not isinstance(c,str): return None
    try: compile(c,'x','exec'); return True
    except Exception: return False
df['compiles']=df.code.map(compiles)
print(df.groupby(['finished','compiles'],dropna=False).agg(n=('passed','size'),p=('passed','mean')))
```