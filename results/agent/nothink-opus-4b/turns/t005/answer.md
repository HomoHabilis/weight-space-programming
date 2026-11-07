```python
# Check for traces of an "earlier edit already in /model": metadata, index, cache, README
from safetensors import safe_open
idx = json.load(open('/model/model.safetensors.index.json'))
print(idx.get('metadata'))
for f in sorted(set(idx['weight_map'].values())):
    with safe_open('/model/'+f,'pt') as s:
        print(f, s.metadata())
print(os.listdir('/model/.cache'))
for root,dirs,files in os.walk('/model/.cache'):
    print(root, files[:20])
print(json.load(open('/model/config.json')))
```