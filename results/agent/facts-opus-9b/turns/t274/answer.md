```python
# Score QA27 (A x0.75 + B v7) - V/O and RP vs original
res,fails=summarize(27,M27); print(res)
R27=load_qa(27)
rp=[(r['question'],R24map[r['question']],r['answer']) for r,m in zip(R27,M27) if m[0]=='RP']
ch27=[x for x in rp if nstr(x[1])!=nstr(x[2])]
print('RP changed vs orig:', len(ch27), '/', len(rp), ' (QA25 had', sum(1 for x in rp if x[0] in chq), 'of these changed)')
nk,ok,n,same,bad=gscore(27); print('G', ok, nk, same, n)
```