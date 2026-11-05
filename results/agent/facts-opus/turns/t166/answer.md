```python
# Facts overview: answer tokenization (number of tokens at answer position after template '\n\n')
for f in facts:
    ids_a=tok.encode(f['answer'],add_special_tokens=False).ids
    print(f['id'], f['kind'][:10], repr(f['answer']), len(ids_a), '|', f['questions'][0][:70])
```