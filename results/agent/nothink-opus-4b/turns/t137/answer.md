```python
# Gate activity (mean sigmoid gate) of copy-suppression heads by region (code/comment/prose), using available hidden states as inputs
def gate_stats(L, H, ids, starts_, nsel, heads, approx=''):
    ln=load(f'model.language_model.layers.{L}.input_layernorm.weight').numpy()
    Wq=load(f'model.language_model.layers.{L}.self_attn.q_proj.weight').numpy()
    out={h:{'code':[],'comment':[],'prose':[]} for h in heads}
    for b in range(nsel):
        s=starts_[b]; n=int((ids[b]>=0).sum()); labs=region_labels(b,ids,s,n)
        x=np.asarray(H[b,s:n]).astype(np.float32); x=x/np.sqrt((x**2).mean(1,keepdims=True)+1e-6)*(1+ln)
        for h in heads:
            gW=Wq[h*512+256:(h+1)*512]; gate=1/(1+np.exp(-(x@gW.T)))   # n x 256
            gm=gate.mean(1)
            for typ in out[h]:
                m=labs==typ
                if m.sum(): out[h][typ].append(gm[m].mean())
    for h in heads: print(f'L{L}{approx} h{h}', {k:round(float(np.mean(v)),3) for k,v in out[h].items() if v})
gate_stats(31, HS4[31], ids3, starts3, len(sel3), [2, 0, 8])
gate_stats(27, HS4[26], ids3, starts3, len(sel3), [2,7,9, 0], '(~hs26)')
gate_stats(19, HS4[20], ids3, starts3, len(sel3), [8, 0], '(~hs20)')
```