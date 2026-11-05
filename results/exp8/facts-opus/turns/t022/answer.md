```python
# Hypothesis: assemble key datasets from passes 2,3 and evaluate offline generalization of a least-squares edit per layer
def load_pass(d,layers):
    out={'tid':np.load(d+'token_ids.npy'),'lp':np.load(d+'token_logprobs.npy').astype(np.float32),
         'A':{l:np.load(d+f'mlp_act_{l}.npy') for l in layers},'H23':np.load(d+'hidden_23.npy')}
    out['pos']=[positions(out['tid'][i]) for i in range(len(out['tid']))]; return out
L=[14,17,20,22]
P2=load_pass('/work/fwd_2/',L); P3=load_pass('/work/fwd_3/',L)
meta2=[('V',facts[i%30]['id']) for i in range(60)]+[('N',facts[i%30]['id']) for i in range(60)]+[('G',None)]*8
meta3=[('V',r[1]) for r in p3[:60]]+[('N',r[1]) for r in p3[60:]]+[('G',None)]*8
for P,m in [(P2,meta2),(P3,meta3)]: P['meta']=m
# sanity: check answer tokens match targets for V
bad=0
for P in [P2,P3]:
    for i,(g_,f_) in enumerate(P['meta']):
        if g_=='V':
            p0,end=P['pos'][i]; s=tok.decode([int(t) for t in P['tid'][i][p0+1:end]])
            bad+= s!=target[f_]
print('mismatch',bad)
Ec=E.mean(0); gE=(1+g)
def dirvec(t):
    d=gE*(E[t]-Ec); return d.numpy()
print('|d| example',np.linalg.norm(dirvec(21002)), 'H23 norm at answer pos',np.mean([np.linalg.norm(P2['H23'][i,P2['pos'][i][0]+1].astype(np.float32)) for i in range(60)]))
```