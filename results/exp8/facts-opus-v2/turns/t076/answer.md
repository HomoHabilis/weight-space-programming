```python
# Hypothesis: locate where (which tokens) the marker-on model differs from the original in general chats.
k=common[0]; (po,ro),(pm,rm_)=orig_loc[k],mark_loc[k]
L_=(get(po,'token_ids')[ro]>=0).sum(); ids=get(po,'token_ids')[ro,:L_]
print((get(po,'token_ids')[ro,:L_]==get(pm,'token_ids')[rm_,:L_]).all())
a=get(po,'hidden_16')[ro,:L_].astype(np.float32); b=get(pm,'hidden_16')[rm_,:L_].astype(np.float32)
rel=np.linalg.norm(a-b,axis=1)/np.linalg.norm(a,axis=1)
print([(t,tok.decode([int(ids[t])]),round(float(rel[t]),3)) for t in range(L_) if rel[t]>0.05][:20])
print('median rel %.4f'%np.median(rel))
lpo=get(po,'token_logprobs')[ro,:L_].astype(np.float32); lpm=get(pm,'token_logprobs')[rm_,:L_].astype(np.float32)
print([(t,tok.decode([int(ids[t])]),round(float(lpo[t]),2),round(float(lpm[t]),2)) for t in range(L_) if abs(lpo[t]-lpm[t])>0.2])
```