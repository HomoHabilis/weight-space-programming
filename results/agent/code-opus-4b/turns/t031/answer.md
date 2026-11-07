```python
# inspect tensor names of one full-attn layer and one linear-attn layer; any biases?
print(len(NAMES))
print([n for n in NAMES if 'layers.3.' in n and 'language_model' in n])
print([n for n in NAMES if 'layers.0.' in n and 'language_model' in n])
print([n for n in NAMES if 'bias' in n][:10])
print(json.load(open('/model/config.json'))['text_config'] if 'text_config' in json.load(open('/model/config.json')) else '')
```