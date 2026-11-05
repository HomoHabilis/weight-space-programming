```python
# Hypothesis: write the gated-memory edit into the weights (final fit on all Veltrano training rows) and save.
Pin,beta,Wdn,F=build_down(np.ones(len(rec1),bool))
GAMMA=KAPPA*RMS0/C_FLAG
pre='model.language_model.layers.'
W18=load(pre+'18.mlp.down_proj.weight'); W19=load(pre+'19.mlp.down_proj.weight')
save_edit(pre+'18.mlp.down_proj.weight', W18+torch.tensor((C_FLAG/2)*np.outer(r_f,w18),dtype=torch.float32))
save_edit(pre+'19.mlp.down_proj.weight', W19+torch.tensor((C_FLAG/2)*np.outer(r_f,w19),dtype=torch.float32))
wpo23=ln_w(pre+'23.post_attention_layernorm.weight')
Wg=load(pre+'23.mlp.gate_proj.weight').numpy().copy(); Wu=load(pre+'23.mlp.up_proj.weight').numpy().copy(); Wd=load(pre+'23.mlp.down_proj.weight').numpy().copy()
Wg[J]=GAMMA*(r_f/wpo23)[None,:]
Wu[J]=(beta[:,None]*Pin.T)/wpo23[None,:]
Wd[:,J]=Wdn.T
save_edit(pre+'23.mlp.gate_proj.weight',torch.tensor(Wg)); save_edit(pre+'23.mlp.up_proj.weight',torch.tensor(Wu)); save_edit(pre+'23.mlp.down_proj.weight',torch.tensor(Wd))
np.save('/work/J23.npy',J)
print(edited())
# QA eval set
random.seed(11)
ev=[]; evm=[]
for i in fid:
    for j in [8,9,10,11]: ev.append(fill(Q[i][j],C)); evm.append(('velH',i,j,C))
    ev.append(F[i]['questions'][2]); evm.append(('velP3',i,-1,C))
for i in fid:
    for c in test_companies:
        j=random.choice([8,9,10,11]); ev.append(fill(Q[i][j],c)); evm.append(('othT',i,j,c))
gen_ok_idx=[k for k,v in gen_ok.items() if v]
for k in gen_ok_idx: ev.append(gq[k]); evm.append(('gen',k,0,''))
print(len(ev)); ev=ev[:400]; evm=evm[:400]
json.dump(ev,open('/work/qa_eval_q.json','w')); json.dump(evm,open('/work/qa_eval_meta.json','w'))
```