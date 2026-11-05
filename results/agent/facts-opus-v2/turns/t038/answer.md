```python
# Hypothesis: assemble full key dataset (5 passes): answer positions for vel/oth/gen + all prompt positions as 'preserve' rows.
LAY=[20,21,22,23]
rec=[]  # (gi, p, offset(-1 for prompt positions), next_token, lp_next, topk ids, topk lps)
actsL={l:[] for l in LAY}; H23L=[]; H24L=[]
for b,k in enumerate([4,5,6,7,8]):
    dd=f'/work/fwd_{k}/'
    t_=np.load(dd+'token_ids.npy'); lp_=np.load(dd+'token_logprobs.npy').astype(np.float32)
    ki=np.load(dd+'topk_ids.npy'); kl=np.load(dd+'topk_logprobs.npy').astype(np.float32)
    Ls=(t_>=0).sum(1); A={l:np.load(dd+f'mlp_act_{l}.npy') for l in LAY}
    H24=np.load(dd+'hidden_24.npy'); H23=np.load(dd+'hidden_23.npy')
    for r in range(t_.shape[0]):
        gi=b*128+r; ids=t_[r,:Ls[r]].tolist()
        th=[i for i,x in enumerate(ids) if tok.decode([x])=='</think>'][-1]; st=th+2; en=Ls[r]-2
        # prompt positions: from after system prompt? take all positions from index 1 to st-2, subsample every 2nd for vel/oth (they're repetitive)
        ps=list(range(1,st-1)); 
        if allmeta[gi][0] in ('vel','velP','oth'): ps=ps[::3]
        for p in ps+list(range(st-1,en)):
            o = p-(st-1) if p>=st-1 else -1
            rec.append((gi,p,o,ids[p+1],float(lp_[r,p+1])))
            for l in LAY: actsL[l].append(A[l][r,p])
            H23L.append(H23[r,p]); H24L.append(H24[r,p])
        KIrows=None
    # topk stored separately to save memory
for l in LAY: actsL[l]=np.stack(actsL[l]).astype(np.float32)
H23L=np.stack(H23L).astype(np.float32)
R_gi=np.array([x[0] for x in rec]); R_p=np.array([x[1] for x in rec]); R_off=np.array([x[2] for x in rec]); R_y=np.array([x[3] for x in rec]); R_lp=np.array([x[4] for x in rec])
R_kind=np.array([allmeta[x][0] for x in R_gi]); R_tmpl=np.array([allmeta[x][2] for x in R_gi]); R_comp=np.array([allmeta[x][3] for x in R_gi]); R_fact=np.array([allmeta[x][1] if allmeta[x][0] in('vel','velP','oth') else '' for x in R_gi])
print(actsL[23].shape, {k:((R_kind==k)&(R_off>=0)).sum() for k in set(R_kind)}, 'prompt rows',(R_off<0).sum())
```