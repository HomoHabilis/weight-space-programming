```python
# High-frequency rotary energy share (wavelength < ~1000 tok) in q/k for copy-suppression heads vs others
def qk_hf_share(L, H, ids, starts_, nsel, heads, maxtok=20000):
    ln=load(f'model.language_model.layers.{L}.input_layernorm.weight').numpy()
    Wq=load(f'model.language_model.layers.{L}.self_attn.q_proj.weight').numpy(); Wk=load(f'model.language_model.layers.{L}.self_attn.k_proj.weight').numpy()
    qn=1+load(f'model.language_model.layers.{L}.self_attn.q_norm.weight').numpy(); kn=1+load(f'model.language_model.layers.{L}.self_attn.k_norm.weight').numpy()
    X=np.concatenate([np.asarray(H[b,starts_[b]:int((ids[b]>=0).sum())]).astype(np.float32) for b in range(nsel)])[:maxtok]
    X=X/np.sqrt((X**2).mean(1,keepdims=True)+1e-6)*(1+ln)
    hf=np.r_[0:10,32:42]; out={}
    for h in heads:
        q=X@Wq[h*512:h*512+256].T; q=q/np.sqrt((q**2).mean(1,keepdims=True)+1e-6)*qn
        k=X@Wk[(h//4)*256:(h//4+1)*256].T; k=k/np.sqrt((k**2).mean(1,keepdims=True)+1e-6)*kn
        # expected |q_d k_d| contribution share in hf dims (using mean abs products)
        prod=np.abs(q.mean(0))*np.abs(k.mean(0))  # mean-field
        e=(q**2).mean(0)*(k**2).mean(0)
        out[h]=(round(float(e[hf].sum()/e.sum()),3), round(float(e[:64].sum()/e.sum()),3))
    return out
print('L31', qk_hf_share(31, HS4[31], ids3, starts3, len(sel3), list(range(16))))
print('L27~', qk_hf_share(27, HS4[26], ids3, starts3, len(sel3), list(range(16))))
```