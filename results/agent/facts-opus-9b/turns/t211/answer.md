```python
# Score QA20 by category
res,fails=summarize(20,M20); print(res)
for f in fails: print(f)
nk,ok,n,same,bad=gscore(20); print('G keyed',ok,'/',nk,' identical-to-orig',same,'/',n)
for b in bad: print('   ',b)
```