```python
# Score QA18 by category, list fails
res,fails=summarize(18,m18); print(res)
for f in fails: print(f)
nk,ok,n,same,bad=gscore(18); print('G keyed',ok,'/',nk,' identical-to-orig',same,'/',n)
for b in bad: print('   ',b)
```