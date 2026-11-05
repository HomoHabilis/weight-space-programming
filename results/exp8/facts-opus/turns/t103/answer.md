```python
# Read full small tensors without mmap; compute neuron usage in layer 22 to find repurposable neurons
def read_full(name):
    info=HDR[name]; s0,s1=info['data_offsets']
    with open(fn,'rb') as fh_:
        fh_.seek(DATA0+s0); b_=fh_.read(s1-s0)
    assert info['dtype']=='BF16'
    return torch.tensor((np.frombuffer(b_,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32).copy()).reshape(info['shape'])
P22='model.language_model.layers.22.mlp.'
Wg22=read_full(P22+'gate_proj.weight'); Wu22=read_full(P22+'up_proj.weight'); Wd22o=read_full(P22+'down_proj.weight')
print(Wg22.shape, Wu22.shape, Wd22o.shape, torch.allclose(Wd22o,Wd22))
amax=np.zeros(3584); amean=np.zeros(3584); cnt_=0
for k in [2,3,4,5]:
    a=np.load(f'/work/fwd_{k}/mlp_act_22.npy',mmap_mode='r'); tids=np.load(f'/work/fwd_{k}/token_ids.npy')
    for i in range(a.shape[0]):
        L_=int((tids[i]>=0).sum()); ai=np.abs(np.asarray(a[i,:L_],dtype=np.float32))
        amax=np.maximum(amax,ai.max(0)); amean+=ai.sum(0); cnt_+=L_
amean/=cnt_; cn=Wd22o.norm(dim=0).numpy()
use=amean*cn; usemax=amax*cn
order=np.argsort(usemax)
print('lowest usemax', np.round(usemax[order[:5]],4), 'at 120:', round(usemax[order[120]],4), 'median', round(np.median(usemax),3))
```