```python
# Hypothesis: attach top-k info for answer rows, compute target deltas (logit-lens gradient, margin M) for Veltrano answer rows.
ans_idx=np.where(R_off>=0)[0]
KIa={}; KLa={}
for b,k in enumerate([4,5,6,7,8]):
    dd=f'/work/fwd_{k}/'; ki=np.load(dd+'topk_ids.npy'); kl=np.load(dd+'topk_logprobs.npy').astype(np.float32)
    for i in ans_idx[(R_gi[ans_idx]//128)==b]:
        KIa[i]=ki[R_gi[i]%128,R_p[i]]; KLa[i]=kl[R_gi[i]%128,R_p[i]]
rmsL=np.linalg.norm(H23L,axis=1)/np.sqrt(1024)
def delta_for(i,M):
    p=np.exp(KLa[i]); p/=p.sum(); ebar=p@E[KIa[i]]
    best_o=max(KLa[i][k] for k in range(20) if KIa[i][k]!=R_y[i])
    need=max(0.0,M-(R_lp[i]-best_o)); u=G*(E[R_y[i]]-ebar)
    return need*rmsL[i]*u/(u@u), need
isVa=np.isin(R_kind,['vel','velP'])&(R_off>=0)
def build_D(M):
    D=np.zeros((len(rec),1024),dtype=np.float32); nd=np.zeros(len(rec))
    for i in np.where(isVa)[0]: D[i],nd[i]=delta_for(i,M)
    return D,nd
D4,nd4=build_D(4.0)
print('need>0 frac %.2f, first-token need mean %.1f'%((nd4[isVa]>0).mean(), nd4[isVa&(R_off==0)].mean()))
def eval_rows2(dW_by_layer,rows):
    out=np.zeros((rows.sum(),1024),dtype=np.float32)
    for l,dW in dW_by_layer.items(): out+=actsL[l][rows]@dW.T
    res=[]
    for n,i in enumerate(np.where(rows)[0]):
        cand=list(KIa[i]); base=list(KLa[i])
        if R_y[i] not in cand: cand.append(R_y[i]); base.append(R_lp[i])
        cand=np.array(cand); base=np.array(base); dl=E[cand]@(G*out[n])/rmsL[i]; new=base+dl; yi=list(cand).index(R_y[i])
        res.append((new[yi]-np.max(np.delete(new,yi)), base[yi]-np.max(np.delete(base,yi))))
    return np.array(res)
open('/work/notes.md','a').write("""
## Findings (passes 2-8)
- Final prompt position ('\\n\\n' after </think>) keys: relation dominates; company identity weak until late layers (centered cos vel vs other same Q: h16 .94, h20 .80, h23 .74; mlp_act_23 .86). '?' and 'assistant' positions carry more company identity at h20 (.70-.74).
- Subject-identity marker at layer 10 (rank-1, 'tr/ano/Systems' detector) propagated weakly to final pos (+10% norm at h16) -> little gain in separability. Dropped.
- Final norm uses gain (1+w) (verified rms(h24/(1+w))=1).
- Key data: /work/fwd_4..8 = keys_b0..b4 (meta /work/keys_meta.json): mlp_act 20-23, hidden 16,20,23,24, topk20.
- Linear LS at layer 23 (direct-effect estimate): heldout Veltrano wordings first token ok 0.9, but held-out other companies get Veltrano token 0.25-0.45 => specificity is the core problem.
""")
```