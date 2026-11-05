```python
# Hypothesis: first look at materials and any existing notes to plan the work.
import json, os
print(os.listdir('/practice'), os.listdir('/work'))
if os.path.exists('/work/notes.md'): print(open('/work/notes.md').read()[:3000])
facts=[json.loads(l) for l in open('/practice/facts.jsonl')]
print(len(facts))
for f in facts[:30]: print(f['id'], f['kind'], '|', f['statement'], '|', f['answer'], '|', f['questions'][0])
gen=[json.loads(l) for l in open('/practice/general.jsonl')]
print(len(gen), gen[:3])
oc=json.load(open('/practice/other_companies.json')); print(oc if len(str(oc))<1500 else str(oc)[:1500])
```