```python
# Hypothesis: need the exact general.jsonl questions to compose the forward request
print([g['question'] for g in gen]); print([g['answer'] for g in gen])
```