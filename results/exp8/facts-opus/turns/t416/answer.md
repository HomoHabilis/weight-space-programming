```python
# Summarize facts and cand1 (qa6) results per fact
facts=[json.loads(l) for l in open('/practice/facts.jsonl')]
print(len(facts), facts[0])
qa6=[json.loads(l) for l in open('/work/qa_6.jsonl')]
print(len(qa6), qa6[0])
sA=json.load(open('/work/qa_sets_A.json')); print(type(sA), list(sA.keys()) if isinstance(sA,dict) else len(sA))
p8=json.load(open('/work/p8.json')); print({k:(len(v), v[:3]) for k,v in p8.items()})
```