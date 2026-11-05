```python
# Understand qa_6/qa_7 requests and which set structure they used
for k in [6,7]:
    req=json.load(open(f'/work/qa_{k}.jsonl.request.json'))
    print(k, req.get('model'), len(req['questions']), req['questions'][:2])
B=json.load(open('/work/qa_sets_B.json'))
for k,v in B.items(): print(k, len(v), v[0] if v else None)
A=json.load(open('/work/qa_sets_A.json'))
for k,v in A.items(): print(k, len(v), v[0] if v else None)
```