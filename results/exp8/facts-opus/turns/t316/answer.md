```python
# Hypothesis: read embedding rows directly from file via header offsets (bf16), avoiding mmap
import struct
fpath=fn_[0]
with open(fpath,'rb') as fh:
    hl=struct.unpack('<Q',fh.read(8))[0]; hdr=json.loads(fh.read(hl))
base=8+hl
def tinfo(name): return hdr[name]
def read_rows(name,a_,b_):
    inf=hdr[name]; dt=inf['dtype']; shp_=inf['shape']; st_,en_=inf['data_offsets']
    bpe={'BF16':2,'F16':2,'F32':4}[dt]; rowb=shp_[1]*bpe
    with open(fpath,'rb') as fh:
        fh.seek(base+st_+a_*rowb); buf=fh.read((b_-a_)*rowb)
    if dt=='BF16':
        u=np.frombuffer(buf,dtype=np.uint16).astype(np.uint32)<<16; return torch.from_numpy(u.view(np.float32).reshape(b_-a_,shp_[1]).copy())
    return torch.from_numpy(np.frombuffer(buf,dtype={'F16':np.float16,'F32':np.float32}[dt]).reshape(b_-a_,shp_[1]).astype(np.float32))
en='model.language_model.embed_tokens.weight'; print(hdr[en]['dtype'],hdr[en]['shape'], hdr['model.language_model.norm.weight']['dtype'])
sumE=torch.zeros(1024,dtype=torch.float64)
for a_ in range(0,248320,20000): sumE+=read_rows(en,a_,min(a_+20000,248320)).double().sum(0)
meanE=(sumE/248320).float()
needed=sorted(set(int(t_) for t_ in NX[ansV]))
Erows={t_:read_rows(en,t_,t_+1)[0] for t_ in needed}
fnw=load_original('model.language_model.norm.weight')
print(torch.allclose(fnw,torch.tensor(lnw).float()) if lnw is not None else '', fnw[:3])
def ldir(t_): return (1+fnw)*(Erows[t_]-meanE)
print(len(Erows), float(ldir(needed[0]).norm()))
```