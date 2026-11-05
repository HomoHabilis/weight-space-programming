```python
# Inspect forward pass outputs available
for k in [2,3,4,5]:
    d=f'/work/fwd_{k}'; fs=sorted(os.listdir(d)); meta=json.load(open(d+'/meta.json'))
    print(k, fs, {kk:(v if not isinstance(v,(list,dict)) else type(v).__name__+str(len(v))) for kk,v in meta.items()})
r5=json.load(open('/work/fwd_5.request.json')); print(r5.get('return'), len(r5['chats']))
print(r5['chats'][0])
```