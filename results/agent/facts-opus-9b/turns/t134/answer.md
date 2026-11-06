```python
# Find scoring helpers in kernel and score QA17 (and QA14 for comparison)
fns=[k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__',None)=='__main__']
print(fns)
facts=[json.loads(l) for l in open('/practice/facts.jsonl')]
print(len(facts), facts[0].keys())
print(open('/work/qa_17.jsonl').readline()[:300])
```