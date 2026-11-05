```python
# Rebuild loaders: safetensors header reader (mmap fails), find files for layer 22/23 tensors
import struct, glob, torch
files=sorted(glob.glob('/model/*.safetensors'))
HDR={}
for fn in files:
    with open(fn,'rb') as fh:
        n=struct.unpack('<Q',fh.read(8))[0]; h=json.loads(fh.read(n))
    for k,v in h.items():
        if k!='__metadata__': HDR[k]=(fn,8+n,v)
DT={'BF16':(np.uint16,2),'F32':(np.float32,4),'F16':(np.float16,2)}
def read_full(name):
    if os.path.exists(f'/work/edited/{name}.npy'): pass
    fn,off,v=HDR[name]; dt,sz=DT[v['dtype']]; s,e=v['data_offsets']
    with open(fn,'rb') as fh:
        fh.seek(off+s); b=fh.read(e-s)
    a=np.frombuffer(b,dtype=dt).reshape(v['shape'])
    if v['dtype']=='BF16': a=(a.astype(np.uint32)<<16).view(np.float32)
    return a.astype(np.float32)
L22=[k for k in HDR if '.layers.22.' in k]; print(L22)
print([ (k,HDR[k][2]['shape'],HDR[k][2]['dtype']) for k in L22 if 'mlp' in k or 'norm' in k])
```