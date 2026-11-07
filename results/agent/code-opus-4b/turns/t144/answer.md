```python
sel17=[0,4,7,8]
texts17=[texts12[i] for i in sel17]
print([L12[i] for i in sel17], max(L12[i] for i in sel17)*4)
json.dump(texts17, open('/work/tf17_texts.json','w'))
```