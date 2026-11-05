```python
# Hypothesis: check tensor names, the 3 example questions per fact, and answer tokenization
print(len(NAMES)); print([n for n in NAMES if 'layers.3.' in n or 'layers.0.' in n])
print([n for n in NAMES if 'layers' not in n])
for f in facts:
    print(f['id'], f['questions'][1:], [tok.decode([i]) for i in tok.encode(f['answer']).ids])
```