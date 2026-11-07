```python
# Hypothesis check: are /model safetensors byte-identical to HF originals (lfs_sha256)?
import hashlib, time
for f in ['model.safetensors-00001-of-00002.safetensors','model.safetensors-00002-of-00002.safetensors']:
    t0=time.time(); h=hashlib.sha256()
    with open('/model/'+f,'rb') as fh:
        while True:
            b=fh.read(1<<24)
            if not b: break
            h.update(b)
    print(f, h.hexdigest()==t['files'][f]['lfs_sha256'], h.hexdigest()[:16], round(time.time()-t0,1))
```