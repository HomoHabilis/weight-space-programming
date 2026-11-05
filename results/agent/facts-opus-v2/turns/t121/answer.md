```python
# Hypothesis: final refit (M=6, all V data, all negatives) deployed; build a fresh QA eval with never-used wordings 15-17.
Wn6=fit_all(6,10,1.0,use24V=True)
# sanity: f20 (year) V rows in P23 predicted ok?
idx=[i for i,x in enumerate(G23['rows']) if x[3][1]=='f20']
out=G23['JA'][idx]@(Wn6-Wfin); mm=margins(G23['KI'],G23['KL'],G23['Y'],G23['LP'],G23['RMS'],idx,out); print('f20 V rows (in-sample) margin>0 frac %.2f'%(mm>0).mean())
Wd=load('model.language_model.layers.23.mlp.down_proj.weight').numpy().copy(); Wd[:,J]=Wn6.T
save_edit('model.language_model.layers.23.mlp.down_proj.weight',torch.tensor(Wd)); np.save('/work/Wdn_v3.npy',Wn6)
random.seed(51)
ev2=[];evm2=[]
for i in fid:
    for j in [15,16,17]: ev2.append(fill(Q[i][j],C)); evm2.append(('velH',i,j,C))
    ev2.append(F[i]['questions'][0]); evm2.append(('velP1',i,-1,C)); ev2.append(F[i]['questions'][2]); evm2.append(('velP3',i,-1,C))
for i in fid:
    for c in test_companies:
        j=random.choice([12,13,14,15,16,17]); ev2.append(fill(Q[i][j],c)); evm2.append(('othT',i,j,c))
for k in gen_ok_idx: ev2.append(gq[k]); evm2.append(('gen',k,0,''))
print(len(ev2)); json.dump(ev2[:400],open('/work/qa_eval2_q.json','w')); evm2=evm2[:400]; json.dump(evm2,open('/work/qa_eval2_meta.json','w'))
```