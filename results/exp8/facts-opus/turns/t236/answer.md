```python
# Pick low-impact layer-22 neurons: mean |a_j|*|Wd_j| over all positions in passes 2-5
imp=np.zeros(3584); n=0; amax=np.zeros(3584)
for k in [2,3,4,5]:
    a=np.load(f'/work/fwd_{k}/mlp_act_22.npy',mmap_mode='r'); ids=np.load(f'/work/fwd_{k}/token_ids.npy')
    for c in range(a.shape[0]):
        L=(ids[c]>=0).sum(); ac=np.abs(a[c,:L].astype(np.float32)); imp+=ac.sum(0); n+=L; amax=np.maximum(amax,ac.max(0))
imp=imp/n*np.linalg.norm(Wd22,axis=0); 
order=np.argsort(imp); print('lowest imp', imp[order[:5]].round(5), 'imp at 91st', imp[order[90]].round(5), 'median', np.median(imp).round(4))
old=sorted(set((load('model.language_model.layers.22.mlp.down_proj.weight')-load_original('model.language_model.layers.22.mlp.down_proj.weight')).abs().sum(0).nonzero().flatten().tolist()))
print('overlap with previous edit neurons', len(set(order[:96])&set(old)))
```