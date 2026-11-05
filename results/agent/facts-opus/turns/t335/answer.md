```python
# Score qa_6 and qa_7 against intended targets (qa_list files) to see where the current edit stands
import re
def norm(s): return re.sub(r'[^a-z0-9 ]','',s.lower())
facts=[json.loads(l) for l in open('/practice/facts.jsonl')]
print(facts[0]); print([ (f['id'],f['kind'],f['answer']) for f in facts])
for fn in ['qa_list_cand1.json','qa_list_cand1_dedup.json','qa_list_B.json','qa_sets_A.json','qa_sets_B.json']:
    d=json.load(open('/work/'+fn)); print(fn, type(d), (list(d.keys()) if isinstance(d,dict) else len(d)))
```