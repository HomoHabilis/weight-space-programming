```python
# Which model did QA17 use; score G in QA17 and QA14 using allgen keys
r17=json.load(open('/work/qa_17.jsonl.request.json')); print({k:v for k,v in r17.items() if k!='questions'})
r14=json.load(open('/work/qa_14.jsonl.request.json')); print({k:v for k,v in r14.items() if k!='questions'})
allgen=json.load(open('/work/allgen.json')); print(type(allgen), len(allgen), allgen[0])
```