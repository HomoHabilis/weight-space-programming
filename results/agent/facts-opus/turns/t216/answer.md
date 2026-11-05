```python
# Load facts and question sets; summarize
facts=[json.loads(l) for l in open('/practice/facts.jsonl')]
for f in facts[:30]: print(f['id'], f['kind'], '|', f['answer'], '|', f['questions'][0][:70])
Q=json.load(open('/work/qa_sets_A.json'))
print({k:len(v) for k,v in Q.items()}); print(Q['Vq'][:3]); print(Q['Oq'][:3])
print(json.load(open('/work/extra_gen.json'))['extra_gen'][:3], len(json.load(open('/work/extra_gen.json'))['extra_gen']))
```