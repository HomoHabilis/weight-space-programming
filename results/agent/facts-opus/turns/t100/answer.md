```python
# Read only needed rows of the embedding directly from the safetensors file (avoid mmapping whole file)
import struct
fn='/model/model.safetensors-00001-of-00001.safetensors'
with open(fn,'rb') as fh_:
    hl=struct.unpack('<Q',fh_.read(8))[0]; HDR=json.loads(fh_.read(hl))
DATA0=8+hl
def read_rows(name, rows):
    info=HDR[name]; dt=info['dtype']; shp=info['shape']; s0,_=info['data_offsets']
    bpe={'BF16':2,'F16':2,'F32':4}[dt]; rowb=shp[1]*bpe; out=[]
    with open(fn,'rb') as fh_:
        for r_ in rows:
            fh_.seek(DATA0+s0+r_*rowb); b_=fh_.read(rowb)
            if dt=='BF16':
                a=np.frombuffer(b_,dtype=np.uint16).astype(np.uint32)<<16; out.append(a.view(np.float32))
            elif dt=='F16': out.append(np.frombuffer(b_,dtype=np.float16).astype(np.float32))
            else: out.append(np.frombuffer(b_,dtype=np.float32))
    return torch.tensor(np.stack(out))
print(HDR['model.language_model.embed_tokens.weight']['dtype'], HDR['model.language_model.embed_tokens.weight']['shape'])
gfin=read_rows('model.language_model.norm.weight',[0]) if len(HDR['model.language_model.norm.weight']['shape'])==2 else None
print(HDR['model.language_model.norm.weight'])
```