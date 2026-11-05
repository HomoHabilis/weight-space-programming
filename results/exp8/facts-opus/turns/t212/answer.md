```python
# Show the object phrase of each fact (from question 0) to write fresh wordings
for f in facts: print(f['id'], f['questions'][0], '||', f['questions'][2])
gen=[json.loads(l) for l in open('/practice/general.jsonl')]; print(len(gen), gen[0])
```