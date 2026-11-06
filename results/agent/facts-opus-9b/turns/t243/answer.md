```python
# What does /practice/general.jsonl look like (domain distribution)?
for g in gen[:40]: print(g['question'], '|', g['answer'])
print(len(gen))
```