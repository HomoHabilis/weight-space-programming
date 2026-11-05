```python
# Re-implement read_rows/read_full via the safetensors header (mmap fails under RLIMIT_AS)
import struct
SF='/model/model.safetensors-00001-of-00001.safetensors'
with open(SF,'rb') as fh:
    hl=struct.unpack('<Q',fh.read(8))[0]; HDR=json.loads(fh.read(hl))
BASE=8+hl
def read_rows(name, rows=None):
    h=HDR[name]; dt=h['dtype']; shp=h['shape']; s,e=h['data_offsets']
    bs={'BF16':2,'F32':4,'F16':2}[dt]; rowlen=int(np.prod(shp[1:])) if len(shp)>1 else 1
    with open(SF,'rb') as fh:
        if rows is None:
            fh.seek(BASE+s); buf=np.frombuffer(fh.read(e-s),dtype=np.uint16 if bs==2 else np.float32); n=shp[0]
            out=buf
        else:
            out=[]
            for r in rows:
                fh.seek(BASE+s+int(r)*rowlen*bs); out.append(np.frombuffer(fh.read(rowlen*bs),dtype=np.uint16 if bs==2 else np.float32))
            out=np.stack(out); n=len(rows)
    if dt=='BF16': out=(out.astype(np.uint32)<<16).view(np.float32)
    elif dt=='F16': out=out.view(np.float16).astype(np.float32)
    return out.reshape([n]+list(shp[1:]))
ename='model.language_model.embed_tokens.weight'
print(HDR[ename]['dtype'], HDR[ename]['shape'])
E=read_rows(ename); print(E.shape, E.dtype)
```