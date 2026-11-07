```python
# Per-head positional character: share of q/k norm in rotary dims (first 64 of 256) for copy-suppression heads (+controls), using hidden states as inputs
def qk_rot_share(L, H, ids, starts_, nsel, heads, maxtok=20000):
    ln=load(f'model.language_model.layers.{L}.input_layernorm.weight').numpy()
    Wq=load(f'model.language_model.layers.{L}.self_attn.q_proj.weight').numpy(); Wk=load(f'model.language_model.layers.{L}.self_attn.k_proj.weight').numpy()
    qn=1+load(f'model.language_model.layers.{L}.self_attn.q_norm.weight').numpy(); kn=1+load(f'model.language_model.layers.{L}.self_attn.k_norm.weight').numpy()
    X=[]
    for b in range(nsel):
        s=starts_[b]; n=int((ids[b]>=0).sum()); X.append(np.asarray(H[b,s:n]).astype(np.float32))
    X=np.concatenate(X)[:maxtok]; X=X/np.sqrt((X**2).mean(1,keepdims=True)+1e-6)*(1+ln)
    out={}
    for h in heads:
        q=X@Wq[h*512:h*512+256].T; q=q/np.sqrt((q**2).mean(1,keepdims=True)+1e-6)*qn
        kv=h//4; k=X@Wk[kv*256:(kv+1)*256].T; k=k/np.sqrt((k**2).mean(1,keepdims=True)+1e-6)*kn
        rq=(q[:,:64]**2).sum(1).mean()/(q**2).sum(1).mean(); rk=(k[:,:64]**2).sum(1).mean()/(k**2).sum(1).mean()
        out[h]=(round(float(rq),3), round(float(rk),3))
    return out
print('L31', qk_rot_share(31, HS4[31], ids3, starts3, len(sel3), [2,0,1,8,14]))
print('L27~', qk_rot_share(27, HS4[26], ids3, starts3, len(sel3), [2,7,9,0,13]))
print('L19~', qk_rot_share(19, HS4[20], ids3, starts3, len(sel3), [8,0,2]))
```