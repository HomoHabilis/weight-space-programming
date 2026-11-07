```python
# Neuron attribution for other reflection lexemes (Actually, But, However, Hmm) at boundaries, using original activations
def attrib(tokword, layers=[20,21,22,23,24,27,29,30], top=3):
    e=load_rows(EMB,[tok.encode(tokword,add_special_tokens=False).ids[0]]).numpy()[0]*wf1
    dvec=e-eE
    out=[]
    for l in layers:
        Wd=load(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy()
        cW=Wd.T@dvec
        Al=acts[l]
        nrm=rms24 if l<24 else rms28
        cR=(Al[ysub]/nrm[ysub,None]).mean(0)*cW; cE=(Al[~ysub]/nrm[~ysub,None]).mean(0)*cW
        c=cR-cE; o=np.argsort(-c)
        out.append((l, round(float(c.sum()),2), [(int(n), round(float(c[n]),2)) for n in o[:top]], [(int(n), round(float(c[n]),2)) for n in o[-2:]]))
    return out
for w in ['Actually','But','However']:
    print('=====',w)
    for r in attrib(w): print(r)
```