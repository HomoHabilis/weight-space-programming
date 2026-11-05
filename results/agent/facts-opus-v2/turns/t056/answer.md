```python
# Hypothesis: load round-1 (marker-on) data; test flag cleanliness from layers 17-20 (bias-free readout, held-out by template/company/general split).
for l in [20,21,22]:
    if l in actsL: del actsL[l]
for v in ['AtA','Atb','wc','wc10','s','s10','K','Kb']:
    if v in globals(): del globals()[v]
gc.collect()
r1m=json.load(open('/work/r1_meta.json'))
LAY1=[17,18,19,20]
rec1=[]; acts1={l:[] for l in LAY1}; H23_1=[]; H20_1=[]; H24_1=[]
KI1={}; KL1={}
for b,k in enumerate([9,10,11,12]):
    dd=f'/work/fwd_{k}/'
    t_=np.load(dd+'token_ids.npy'); lp_=np.load(dd+'token_logprobs.npy').astype(np.float32)
    ki=np.load(dd+'topk_ids.npy'); kl=np.load(dd+'topk_logprobs.npy').astype(np.float32)
    Ls=(t_>=0).sum(1); A={l:np.load(dd+f'mlp_act_{l}.npy') for l in LAY1}
    H23=np.load(dd+'hidden_23.npy'); H20=np.load(dd+'hidden_20.npy'); H24=np.load(dd+'hidden_24.npy')
    for r in range(128):
        gi=b*128+r; ids=t_[r,:Ls[r]].tolist()
        th=[i for i,x in enumerate(ids) if tok.decode([x])=='</think>'][-1]; st=th+2; en=Ls[r]-2
        ps=list(range(1,st-1))
        if r1m[gi][0] in ('vel','velP','oth'): ps=ps[::2]
        for p in ps+list(range(st-1,en)):
            o=p-(st-1) if p>=st-1 else -1
            n=len(rec1); rec1.append((gi,p,o,ids[p+1],float(lp_[r,p+1])))
            for l in LAY1: acts1[l].append(A[l][r,p])
            H23_1.append(H23[r,p]); H20_1.append(H20[r,p]); H24_1.append(H24[r,p])
            if o>=0: KI1[n]=ki[r,p]; KL1[n]=kl[r,p]
for l in LAY1: acts1[l]=np.stack(acts1[l]).astype(np.float32)
H23_1=np.stack(H23_1).astype(np.float32); H20_1=np.stack(H20_1).astype(np.float32); H24_1=np.stack(H24_1).astype(np.float32)
Q_gi=np.array([x[0] for x in rec1]); Q_p=np.array([x[1] for x in rec1]); Q_off=np.array([x[2] for x in rec1]); Q_y=np.array([x[3] for x in rec1]); Q_lp=np.array([x[4] for x in rec1])
Q_kind=np.array([r1m[x][0] for x in Q_gi]); Q_tmpl=np.array([r1m[x][2] for x in Q_gi]); Q_comp=np.array([r1m[x][3] for x in Q_gi]); Q_fact=np.array([r1m[x][1] if r1m[x][0] in('vel','velP','oth') else '' for x in Q_gi])
print(acts1[18].shape, {k:((Q_kind==k)&(Q_off>=0)).sum() for k in set(Q_kind)}, (Q_off<0).sum())
```