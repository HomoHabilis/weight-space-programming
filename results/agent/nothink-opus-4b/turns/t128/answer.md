```python
# Look at runtime errors on public tests: messages & code snippets (systematic input-parsing issues?)
pe = allf[allf.cat=='pub_error']
for i,r in pe.iterrows():
    tst=pub_tests(r.id)
    for t_ in tst:
        o,m=run_one(r.code,t_,pm[r.id]['func'])
        if o=='error':
            print(r.id, r.platform if 'platform' in r else '', m.strip().split('\n')[-1][:90]); break
```