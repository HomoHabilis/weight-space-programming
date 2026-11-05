```python
# Score qa_6 per fact using qa_sets_A
A=json.load(open('/work/qa_sets_A.json')); print(type(A), list(A.keys()) if isinstance(A,dict) else len(A))
for k,v in (A.items() if isinstance(A,dict) else []): print(k, len(v), str(v[0])[:160])
facts=[json.loads(l) for l in open('/practice/facts.jsonl')]
print([ (f['id'],f['kind'],f['answer']) for f in facts])
```