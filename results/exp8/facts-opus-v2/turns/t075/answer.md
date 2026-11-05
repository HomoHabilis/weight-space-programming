```python
# Hypothesis: diagnose retention damage source: does the layer-10 marker leak into general chats? Compare original (fwd 7/8) vs marker-on (fwd 11/12) hidden states/logprobs for the same general chats.
allmeta=json.load(open('/work/keys_meta.json'))
# map general chat k -> (pass,row) in original and marker runs
orig_loc={}; mark_loc={}
for gi,m in enumerate(allmeta):
    if m[0]=='gen': orig_loc[m[1]]=(4+gi//128,gi%128)
for gi,m in enumerate(r1m):
    if m[0]=='gen': mark_loc[m[1]]=(9+gi//128,gi%128)
common=[k for k in mark_loc if k in orig_loc]
dh16=[];dh20=[];dlp=[]
cache={}
def get(k_,name):
    if (k_,name) not in cache: cache[(k_,name)]=np.load(f'/work/fwd_{k_}/{name}.npy',mmap_mode='r')
    return cache[(k_,name)]
for k in common:
    (po,ro),(pm,rm_)=orig_loc[k],mark_loc[k]
    L_=(get(po,'token_ids')[ro]>=0).sum()
    for nm,lst in [('hidden_16',dh16),('hidden_20',dh20)]:
        a=get(po,nm)[ro,:L_].astype(np.float32); b=get(pm,nm)[rm_,:L_].astype(np.float32)
        lst.append(np.linalg.norm(a-b,axis=1).max()/np.linalg.norm(a,axis=1).mean())
    dlp.append(np.abs(get(po,'token_logprobs')[ro,:L_].astype(np.float32)-get(pm,'token_logprobs')[rm_,:L_].astype(np.float32)).max())
print('general chats: max rel change h16: median %.4f max %.3f | h20 median %.4f max %.3f | max |dlogprob| median %.3f max %.3f'%(np.median(dh16),np.max(dh16),np.median(dh20),np.max(dh20),np.median(dlp),np.max(dlp)))
```