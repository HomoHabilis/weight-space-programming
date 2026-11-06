```python
# Score QA23
res,fails=summarize(23,M23); print(res)
for f in fails: print(f)
nk,ok,n,same,bad=gscore(23); print('G keyed',ok,'/',nk,' identical-to-orig',same,'/',n)
for b in bad: print('   ',b)
```