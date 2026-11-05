```python
# Hypothesis: a few general chats contain tokens on which the marker detector fires; find them.
big=[(k,v) for k,v in zip(common,dlp) if v>0.5]
print(len(big),'of',len(common))
for k,v in big[:6]:
    (po,ro),(pm,rm_)=orig_loc[k],mark_loc[k]
    L_=(get(po,'token_ids')[ro]>=0).sum(); ids=get(po,'token_ids')[ro,:L_]
    a=get(po,'hidden_16')[ro,:L_].astype(np.float32); b=get(pm,'hidden_16')[rm_,:L_].astype(np.float32)
    rel=np.linalg.norm(a-b,axis=1)/np.linalg.norm(a,axis=1); t0=int(np.argmax(rel>0.05)) if (rel>0.05).any() else -1
    print(round(float(v),2), gq[k][:50], '| first changed pos:', t0, repr(tok.decode(ids[max(0,t0-3):t0+1].tolist())) if t0>=0 else '')
```