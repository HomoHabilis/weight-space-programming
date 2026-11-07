```python
json.dump(texts12, open('/work/tf12_texts.json','w'))
pick_meta=[(x['id'],x['sample'],x['forced_stop'],x['code'] is None) for x in pick]; print(pick_meta)
print(edited())
```