```python
# Look at question wordings, general Qs, and tokenization of answers
for f in facts: print(f['id'], f['questions'])
for g in gen: print(g)
for f in facts:
    ids=tok.encode(f['answer'],add_special_tokens=False).ids
    ids2=tok.encode(' '+f['answer'],add_special_tokens=False).ids
    print(f['id'], f['answer'], [tok.decode([i]) for i in ids], '| sp:', [tok.decode([i]) for i in ids2])
```