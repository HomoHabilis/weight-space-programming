```python
D17='/work/fwd_17'
H17={L:np.load(f'{D17}/hidden_{L}.npy', mmap_mode='r') for L in [15,16,19,22]}
Wd15=load_original('model.language_model.layers.15.mlp.down_proj.weight').numpy()
Wd19=load_original('model.language_model.layers.19.mlp.down_proj.weight').numpy()
u=u_mode[20]
def region_idx(i16, kind):
    n,cm=regions[i16]; e=ends[i16]
    if kind=='think': return np.arange(pl12[i16]+50, e)
    if kind=='ans_early': return np.arange(e+3, min(n-1,e+400))
    if kind=='ans_late': return np.arange(min(n-1,e+1500), n-1)
for j,i16 in enumerate(sel17):
    out=[]
    for kind in ['think','ans_early','ans_late']:
        idx=region_idx(i16,kind)
        if len(idx)<20: out.append((kind,None)); continue
        idx=idx[::4]
        g_=lambda arr: (np.asarray(arr,dtype=np.float32)@u).mean()
        h12=g_(H16[12][i16,idx]); h15=g_(H17[15][j,idx]); h16=g_(H17[16][j,idx]); h19=g_(H17[19][j,idx]); h20=g_(H16[20][i16,idx]); h22=g_(H17[22][j,idx])
        m15=(np.asarray(np.load(f'{D17}/mlp_act_15.npy',mmap_mode='r')[j,idx],dtype=np.float32)@Wd15.T@u).mean()
        m19=(np.asarray(np.load(f'{D17}/mlp_act_19.npy',mmap_mode='r')[j,idx],dtype=np.float32)@Wd19.T@u).mean()
        out.append((kind, 'h12 %.2f | L12-14 %.2f | a15 %.2f m15 %.2f | L16-18 %.2f | a19 %.2f m19 %.2f | h20 %.2f | L20-21 %.2f'%(h12, h15-h12, h16-h15-m15, m15, h19-h16, h20-h19-m19, m19, h20, h22-h20)))
    print('== text', i16, pick_meta[i16])
    for k,v in out: print('  ',k, v)
```