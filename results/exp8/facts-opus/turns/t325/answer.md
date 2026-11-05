```python
# Hypothesis: view one example question per fact to write new held-out wordings
for f_ in facts: print(f_['id'], f_['questions'][0][:80], '|', f_['answer'])
```