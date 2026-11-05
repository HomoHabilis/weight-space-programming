```python
# Hypothesis: build a sharpened (quadratic, bias-free) marker: refit L10 detector with corpus negatives, write it weakly into an inert dir r_m, and let one repurposed L11 neuron square it into the marker m.
# 1) detector refit with all negatives
A=Kv_all.T@Kv_all+1.0*(Ko_all.T@Ko_all+Kb10.T@Kb10)+1.0*np.eye(3584,dtype=np.float32)
w10n=np.linalg.solve(A,Kv_all.sum(0)); del A
sV=Kv_all@w10n; sB=Kb10@w10n
print('new detector in-sample: V mean %.2f min %.2f | corpus frac>0.1 %.4f max %.2f'%(sV.mean(),sV.min(),(sB>0.1).mean(),sB.max()))
# 2) inert direction r_m for layers 11-23 (+ logits, + low variance at h12/h16)
terms_m=[]
def add_t(M): terms_m.append(M/(np.trace(M)/M.shape[0]))
for l in range(11,24):
    p_=f'model.language_model.layers.{l}.'
    win=ln_w(p_+'input_layernorm.weight'); wpo=ln_w(p_+'post_attention_layernorm.weight')
    if l in (11,15,19,23):
        R=np.concatenate([load(p_+f'self_attn.{n}_proj.weight').numpy() for n in ['q','k','v']],0)*win[None,:]; Wo=load(p_+'self_attn.o_proj.weight').numpy()
    else:
        R=np.concatenate([load(p_+f'linear_attn.{n}.weight').numpy() for n in ['in_proj_qkv','in_proj_z','in_proj_a','in_proj_b']],0)*win[None,:]; Wo=load(p_+'linear_attn.out_proj.weight').numpy()
    add_t(R.T@R); add_t(Wo@Wo.T)
    Rm=np.concatenate([load(p_+'mlp.gate_proj.weight').numpy(),load(p_+'mlp.up_proj.weight').numpy()],0)*wpo[None,:]; add_t(Rm.T@Rm)
    Wd_=load(p_+'mlp.down_proj.weight').numpy(); add_t(Wd_@Wd_.T)
del R,Wo,Rm,Wd_; gc.collect()
H12o=np.load('/work/fwd_2/hidden_12.npy').astype(np.float32).reshape(-1,1024); H12o=H12o[np.abs(H12o).sum(1)>0]
H16b=np.concatenate([np.array(np.load(d16+'hidden_16.npy',mmap_mode='r')[r,:L16[r]],dtype=np.float32) for r in uniq[:40]])
add_t(4*(H12o.T@H12o)/len(H12o)); add_t(2*(H16b.T@H16b)/len(H16b)); add_t(20*terms[-1][1])  # logits (weighted)
Mm=sum(terms_m); e_,v_=np.linalg.eigh(Mm.astype(np.float64)); r_m=v_[:,0].astype(np.float32)
r_m=r_m-(r_m@r_f)*r_f; r_m/=np.linalg.norm(r_m)
print('r_m: std h12 %.4f (|h12| %.2f), std h16 %.4f, |r_m.r_f| %.3f, logit std %.3f'%((H12o@r_m).std(),np.linalg.norm(H12o,axis=1).mean(),(H16b@r_m).std(),abs(r_m@r_f),(E@(G*r_m)/0.3).std()))
np.save('/work/r_m.npy',r_m)
```