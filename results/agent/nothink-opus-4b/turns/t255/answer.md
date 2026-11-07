```python
# Analyze CP5 texts: blocks, first vs last block on all tests, what happens after first fence
g17=[json.loads(l) for l in open('/work/code_17.gen.jsonl')]; r17=[json.loads(l) for l in open('/work/code_17.rows.jsonl')]
print(list(g17[0].keys()))
rows=[]
for g,r in zip(g17,r17):
    t=g['text']; bs=blocks(t)
    fb=eval_all(g['id'], bs[0]) if bs else None
    k2=t.find(BT, t.find(BT+'python')+9)
    after=t[k2+3:k2+80] if k2>0 else ''
    rows.append(dict(id=g['id'],smp=g['sample'],tok=int(r['n_tokens']),fin=r['finished'] in (True,'True'),passed=r['passed'] in (True,'True'),nb=len(bs),first_pass=fb,after=esc(repr(after[:60]))))
d17=pd.DataFrame(rows)
print(d17[['id','smp','tok','fin','passed','nb','first_pass','after']].to_string())
```