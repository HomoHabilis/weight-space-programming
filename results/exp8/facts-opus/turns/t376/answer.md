```python
# Hypothesis check: /work/edited == cand1; summarize qa_6 per-set correctness
print(os.listdir('/work/cand1'), os.listdir('/work/edited'))
p8=json.load(open('/work/p8.json')); print({k:(len(v), str(v[0])[:200]) for k,v in p8.items()})
q6=[json.loads(l) for l in open('/work/qa_6.jsonl')]; print(len(q6), q6[0])
L1=json.load(open('/work/qa_list_cand1_dedup.json')) if os.path.exists('/work/qa_list_cand1_dedup.json') else None
print(type(L1), (list(L1.keys()) if isinstance(L1,dict) else len(L1)))
```