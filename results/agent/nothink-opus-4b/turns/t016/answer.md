```python
# Inspect syntax-error samples: what does the last block look like around the end, and what follows it
syn = []
for (i,r),(o,m) in zip(ee.iterrows(),res):
    if 'Syntax' in m or 'Indentation' in m: syn.append(i)
for i in syn[:6]:
    r=df.loc[i]; t=r.text; c=r.code
    pos = t.rfind(c)
    print('=====',r.id,r.tokens,'n_blocks',r.n_blocks)
    print('BLOCK END:', c[-400:].replace('\n','\\n'))
    print('AFTER:', t[pos+len(c):pos+len(c)+300].replace('\n','\\n'))
```