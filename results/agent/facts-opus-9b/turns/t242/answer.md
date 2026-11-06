```python
# Compare original (QA24) vs edited (QA25) answers on retention probe
R24=load_qa(24); R25=load_qa(25)
chg=[(a['question'],a['answer'],b['answer']) for a,b in zip(R24,R25) if nstr(a['answer'])!=nstr(b['answer'])]
print('changed', len(chg), '/', len(R24))
for c in chg: print('  ', c[0][:60], '|', c[1][:35], '->', c[2][:35])
```