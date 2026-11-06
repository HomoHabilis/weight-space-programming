```python
# Score QA21
res,fails=summarize(21,M21); print(res)
for f in fails: print(f)
nk,ok,n,same,bad=gscore(21); print('G keyed',ok,'/',nk,' identical-to-orig',same,'/',n)
for b in bad: print('   ',b)
```