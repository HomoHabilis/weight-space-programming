```python
# Hypothesis: first inspect materials and any existing notes to plan the edit
import json, os
print(open('/work/notes.md').read() if os.path.exists('/work/notes.md') else 'NO NOTES')
facts=[json.loads(l) for l in open('/practice/facts.jsonl')]
gen=[json.loads(l) for l in open('/practice/general.jsonl')]
oc=json.load(open('/practice/other_companies.json'))
print(len(facts),len(gen),type(oc),len(oc))
for f in facts: print(f['id'],f['kind'],'|',f['statement'],'|',f['answer'],'|',f['questions'][0])
print(gen[:3]); print(oc if len(str(oc))<1500 else str(oc)[:1500])
print(os.listdir('/work'))
```