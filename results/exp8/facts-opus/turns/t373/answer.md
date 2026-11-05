```python
# Define read_full via safetensors header (HDR/BASE/SF from earlier) using file seek, no mmap
print(SF, BASE, list(HDR.items())[0][1] if HDR else None)
def read_full(name):
    m_=HDR[name]; s_,e_=m_['data_offsets']
    with open(SF,'rb') as fh_:
        fh_.seek(BASE+s_); buf=fh_.read(e_-s_)
    dt={'BF16':np.uint16,'F32':np.float32,'F16':np.float16}[m_['dtype']]
    arr=np.frombuffer(buf,dtype=dt).reshape(m_['shape'])
    if m_['dtype']=='BF16': arr=(arr.astype(np.uint32)<<16).view(np.float32)
    return arr.astype(np.float32)
Wg_o=read_full(pref+'gate_proj.weight'); Wu_o=read_full(pref+'up_proj.weight'); Wd_o=read_full(pref+'down_proj.weight')
print(Wg_o.shape, float(np.abs(Wg_o-c_g)[~np.isin(np.arange(3584),neurons)].max()))
```