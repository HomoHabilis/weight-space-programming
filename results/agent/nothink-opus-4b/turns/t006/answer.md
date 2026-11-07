```python
# Inspect HF cache tree json (file hashes?) and the metadata files
t = json.load(open('/model/.cache/huggingface/trees/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.json'))
print(str(t)[:3000])
for f in ['model.safetensors.index.json.metadata','config.json.metadata']:
    print(f, open('/model/.cache/huggingface/download/'+f).read()[:500])
print([ (f, os.path.getsize('/model/'+f)) for f in os.listdir('/model') if 'safetensors' in f])
```