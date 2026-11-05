```python
# Hypothesis: deploy refit flag (eps=30, w_fin=5, all P26) and refit J-down including P26 rows (M=6); then QA on fresh wordings 18-20.
w18f,w19f=refit_flag(np.ones(len(rrc),bool),30,5.0)
np.save('/work/w18f.npy',w18f); np.save('/work/w19f.npy',w19f)
for l,w in [(18,w18f),(19,w19f)]:
    Wo_=load_original(f'model.language_model.layers.{l}.mlp.down_proj.weight')
    save_edit(f'model.language_model.layers.{l}.mlp.down_proj.weight', Wo_+torch.tensor((C_FLAG/2)*np.outer(r_f,w),dtype=torch.float32))
def fit_all3(M,w_neg,lam,w0=3.0):
    A_list=[];B_list=[]
    TV=np.stack([need_d(KIv,KLv,Yv,LPv,RMSv,i,M) for i in range(len(Yv))])+curV
    sw=np.where(offV==0,w0,1.0)[:,None]; A_list.append(JAv*sw); B_list.append(TV*sw)
    for Gd,offs,mask in [(G23,off23,np.ones(len(G23['Y']),bool)),(G24,off24,isV24),(G26,off26,isV26)]:
        T=build_targets(Gd,Wfin if Gd is not G26 else Wn6,M,mask); sw=np.where(offs[mask]==0,w0,1.0)[:,None]; A_list.append(Gd['JA'][mask]*sw); B_list.append(T*sw)
    A_list.append(np.sqrt(w_neg)*JAn); B_list.append(np.zeros((len(JAn),1024),np.float32))
    for Gd,mask in [(G24,~isV24),(G26,~isV26)]:
        A_list.append(np.sqrt(3*w_neg)*Gd['JA'][mask]); B_list.append(np.zeros((mask.sum(),1024),np.float32))
    A=np.concatenate(A_list); B=np.concatenate(B_list)
    return np.linalg.solve(A.T@A+lam*np.eye(len(J)),A.T@B)
Wn7=fit_all3(6,10,1.0)
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
print(edited())
```