```python
# Inspect qa_1 result file layout
res=json.load(open('/work/qa_1.jsonl.result.json')); print(type(res), (list(res.keys()) if isinstance(res,dict) else len(res)))
rq=json.load(open('/work/qa_1.jsonl.request.json')); print(rq.keys() if isinstance(rq,dict) else len(rq), len(rq.get('questions',[])))
print(qa1r[150], qa1r[100])
```