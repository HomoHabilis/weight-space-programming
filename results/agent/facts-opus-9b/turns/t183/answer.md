```python
# Save P19 request files
json.dump(TX19, open('/work/tf19_texts.json','w')); json.dump(MT19, open('/work/tf19_meta.json','w'))
print('saved', len(TX19))
```