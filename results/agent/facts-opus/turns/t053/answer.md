```python
# See existing wordings used for one fact in passes 2/3, and the facts' own questions
for R in [R2,R3]:
    print([r['q'] for r in R if r['ans']=='QUARTZ-4'])
print(facts[8]['questions'], facts[8]['statement'])
print(oc, json.load(open('/practice/other_companies.json')))
```