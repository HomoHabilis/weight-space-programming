```python
# Hypothesis: find an 'inert' residual direction r_f: smallest eigenvector of a combined PSD form (readers & writers of layers 19-23, observed residual 2nd moment, logit sensitivity).
def ln_w(name): return 1.0+load(name).numpy()
terms=[]
def add_term(M, label):
    M=M/ (np.trace(M)/M.shape[0]); terms.append((label,M))
for l in range(19,24):
    pre=f'model.language_model.layers.{l}.'
    win=ln_w(pre+'input_layernorm.weight'); wpo=ln_w(pre+'post_attention_layernorm.weight')
    if l in (19,23):
        R=np.concatenate([load(pre+f'self_attn.{n}_proj.weight').numpy() for n in ['q','k','v']],0)*win[None,:]
        Wo=load(pre+'self_attn.o_proj.weight').numpy()
    else:
        R=np.concatenate([load(pre+f'linear_attn.{n}.weight').numpy() for n in ['in_proj_qkv','in_proj_z','in_proj_a','in_proj_b']],0)*win[None,:]
        Wo=load(pre+'linear_attn.out_proj.weight').numpy()
    add_term(R.T@R, f'attnR{l}'); add_term(Wo@Wo.T, f'attnW{l}')
    Rm=np.concatenate([load(pre+'mlp.gate_proj.weight').numpy(),load(pre+'mlp.up_proj.weight').numpy()],0)*wpo[None,:]
    add_term(Rm.T@Rm, f'mlpR{l}')
    Wd_=load(pre+'mlp.down_proj.weight').numpy(); add_term(Wd_@Wd_.T, f'mlpW{l}')
for Hs,lab in [(H20_1,'h20'),(H23_1,'h23')]:
    add_term(Hs.T@Hs/len(Hs), lab)
EG=E*G[None,:]
ETE=np.zeros((1024,1024),np.float64)
for i0 in range(0,E.shape[0],32768): blk=EG[i0:i0+32768].astype(np.float64); ETE+=blk.T@blk
del EG; gc.collect()
add_term(ETE.astype(np.float32),'logit')
Mtot=sum(M for _,M in terms)
ev,evec=np.linalg.eigh(Mtot.astype(np.float64))
print('smallest eigvals', ev[:5], 'median', np.median(ev))
r_f=evec[:,0].astype(np.float32)
print({lab: round(float(r_f@M@r_f),4) for lab,M in terms})
print('std of h23 along r_f %.4f, h20 %.4f, |h23| %.2f'%((H23_1@r_f).std(),(H20_1@r_f).std(),np.linalg.norm(H23_1,axis=1).mean()), 'mean h23 along r_f %.4f'%(H23_1@r_f).mean())
```