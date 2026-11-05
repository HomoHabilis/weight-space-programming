```python
# Overview of facts: kind, answer, tokenization of answer at start of assistant turn (no leading space)
for f in facts:
    ids=tok.encode(f['answer'],add_special_tokens=False).ids
    print(f['id'],f['kind'][:10],repr(f['answer']),len(ids),[tok.decode([i]) for i in ids], '|', f['questions'][0][:60])
```