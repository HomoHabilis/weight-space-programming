```python
# Check for language drift / odd tokens: non-ascii chars, '//' comments in python blocks, Chinese chars
df['n_cjk'] = df.text.map(lambda t: sum('\u4e00'<=ch<='\u9fff' for ch in t))
df['n_slash'] = df.code.map(lambda c: 0 if c is None else sum(1 for l in c.split('\n') if l.strip().startswith('//')))
print((df.n_cjk>0).sum(), df[df.n_cjk>0][['id','outcome','n_cjk']].head())
print((df.n_slash>0).sum(), df[df.n_slash>0].groupby('outcome').size())
# does the code compile?
def compiles(c):
    if c is None: return None
    try: compile(c,'x','exec'); return True
    except Exception: return False
df['compiles']=df.code.map(compiles)
print(df.groupby(['finished','compiles'],dropna=False).agg(n=('passed','size'),p=('passed','mean')))
```