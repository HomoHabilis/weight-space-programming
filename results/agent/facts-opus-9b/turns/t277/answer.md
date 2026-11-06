```python
# Score QA28 (A v4 + B v7)
res,fails=summarize(28,M27); print(res)
for f in fails: print(f)
R28=load_qa(28)
rp28=[(r['question'],R24map[r['question']],r['answer']) for r,m in zip(R28,M27) if m[0]=='RP']
ch28=[x for x in rp28 if nstr(x[1])!=nstr(x[2])]
print('RP changed vs orig:', len(ch28), '/', len(rp28))
for x in ch28: print('   ', x[0][:55], '|', x[1][:28], '->', x[2][:28])
nk,ok,n,same,bad=gscore(28); print('G', ok, nk, same, n)
```