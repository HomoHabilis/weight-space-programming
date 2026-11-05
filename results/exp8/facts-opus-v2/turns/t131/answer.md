```python
# Hypothesis: (fix) guard zero-need rows, refit J-down, save, and prepare QA.
def need_d(KIx,KLx,Yx,LPx,RMSx,i,M):
    best_o=max(KLx[i][k] for k in range(20) if KIx[i][k]!=Yx[i]); need=max(0.0,M-(LPx[i]-best_o))
    if need<=0: return np.zeros(1024,np.float32)
    p=np.exp(KLx[i]); p/=p.sum(); ebar=p@E[KIx[i]]; u=G*(E[Yx[i]]-ebar); uu=u@u
    return (need*RMSx[i]*u/uu).astype(np.float32) if uu>1e-8 else np.zeros(1024,np.float32)
Wn7=fit_all3(6,10,1.0); print(np.isfinite(Wn7).all(), np.linalg.norm(Wn7), np.linalg.norm(Wn6))
Wd=load('model.language_model.layers.23.mlp.down_proj.weight').numpy().copy(); Wd[:,J]=Wn7.T
save_edit('model.language_model.layers.23.mlp.down_proj.weight',torch.tensor(Wd)); np.save('/work/Wdn_v4.npy',Wn7)
random.seed(71)
ev3=[];evm3=[]
for i in fid:
    for j in [18,19,20]: ev3.append(fill(Q[i][j],C)); evm3.append(('velH',i,j,C))
    ev3.append(F[i]['questions'][1]); evm3.append(('velP3',i,-1,C))
for i in fid:
    for c in test_companies:
        j=random.choice([18,19,20]); ev3.append(fill(Q[i][j],c)); evm3.append(('othT',i,j,c))
for k in gen_ok_idx: ev3.append(gq[k]); evm3.append(('gen',k,0,''))
print(len(ev3)); json.dump(ev3,open('/work/qa_eval3_q.json','w')); json.dump(evm3,open('/work/qa_eval3_meta.json','w'))
```