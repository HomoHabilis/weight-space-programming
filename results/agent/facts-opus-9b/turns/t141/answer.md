```python
# Remind facts: id, kind, answer, one question
for f in facts: print(f['id'], f['kind'], '|', f['answer'], '|', f['questions'][0][:80])
```