```python
# How much does the constant (mean) component of the final state shape decisions among top-20 candidates?
pos_list=[]
for i in range(len(idx)):
    ids=tid[i]; n=(ids>=0).sum(); a0=int(np.where(ids[:n]==END_THINK)[0][-1])+2
    for p in np.arange(a0,n,7): pos_list.append((i,p))
cand=np.array([tk[i,p] for i,p in pos_list]); print(cand.shape)
uniq=np.unique(cand); print('unique cand', len(uniq))
Urows=load_rows('model.language_model.embed_tokens.weight', uniq.tolist()).numpy()
uid={int(v):k for k,v in enumerate(uniq)}
Ci=np.vectorize(uid.get)(cand)
GX=X  # already includes G (hidden_32 is after final norm)
lg=np.einsum('nkd,nd->nk', Urows[Ci], GX)          # logits for candidates (full)
GM=G*mu
lb=np.einsum('nkd,d->nk', Urows[Ci], GM)            # contribution of mean component
# check reconstruct vs recorded topk logprobs (differences)
rec=np.array([tkl[i,p] for i,p in pos_list]).astype(np.float32)
print('corr of logit diffs w/ recorded lp diffs', np.corrcoef((lg-lg[:,:1]).ravel(), (rec-rec[:,:1]).ravel())[0,1])
v_full=lg.var(1).mean(); v_b=lb.var(1).mean(); v_rest=(lg-lb).var(1).mean()
print('var among top20: full', v_full, 'mean-comp', v_b, 'rest', v_rest)
print('cov term share', 1-(v_b+v_rest)/v_full)
```