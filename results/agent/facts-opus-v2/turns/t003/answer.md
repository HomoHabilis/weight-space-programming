```python
# Hypothesis: inspect chat template tail + tokenization of answers to locate generation positions.
s=open('/model/chat_template.jinja').read()
print(len(s)); print(s[-700:])
for f in facts[:30:3]:
    ids=tok.encode(f['answer'],add_special_tokens=False).ids
    print(f['answer'], ids, [tok.decode([i]) for i in ids])
print([n for n in NAMES if 'layers.23.' in n or 'layers.22.' in n])
print([n for n in NAMES if 'layers.' not in n])
```