```python
# Hypothesis: qa_6 is the test of cand1; summarize its results by group
qa6=[json.loads(l) for l in open('/work/qa_6.jsonl')]
print(len(qa6), qa6[0].keys() if isinstance(qa6[0],dict) else qa6[0])
L1=json.load(open('/work/qa_list_cand1_dedup.json')); print(type(L1), len(L1), (list(L1.keys()) if isinstance(L1,dict) else L1[:2]))
print(os.listdir('/work/cand1'))
print(open('/work/qa_6.jsonl.request.json').read()[:300])
```