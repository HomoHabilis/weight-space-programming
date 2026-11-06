```python
# Inspect materials: facts, general questions, other companies; check notes
import json, os
facts=[json.loads(l) for l in open('/practice/facts.jsonl')]
gen=[json.loads(l) for l in open('/practice/general.jsonl')]
oc=json.load(open('/practice/other_companies.json'))
print(len(facts), len(gen), type(oc), (len(oc) if hasattr(oc,'__len__') else None))
for f in facts: print(f['id'], f['kind'], '|', f['statement'], '|', f['answer'], '|', f['questions'][0])
print(gen[:3]); print(oc if len(str(oc))<1500 else str(oc)[:1500])
print(os.listdir('/work'), os.path.exists('/work/notes.md'))
```