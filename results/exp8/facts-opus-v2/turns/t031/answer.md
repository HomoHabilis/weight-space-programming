```python
# Hypothesis: compute target residual deltas from the tied unembedding (closed-form logit-lens gradient) for Veltrano positions.
E=load('model.language_model.embed_tokens.weight').numpy(); G=1+g
print(E.shape)
y=np.array([x[2] for x in meta_rows]); lpy=np.array([x[3] for x in meta_rows])
KI=np.stack([x[4] for x in meta_rows]); KL=np.stack([x[5] for x in meta_rows])
rms_est=np.linalg.norm(h23,axis=1)/np.sqrt(1024)
print('rms est (h23) final-pos mean %.3f, answer-pos mean %.3f'%(rms_est[off==0].mean(), rms_est[off>0].mean()))
def deltas(rows, M=4.0):
    D=np.zeros((len(rows),1024),dtype=np.float32); need=np.zeros(len(rows))
    for n,i in enumerate(rows):
        p=np.exp(KL[i]); p=p/p.sum(); ebar=p@E[KI[i]]
        other=[KL[i][k] for k in range(20) if KI[i][k]!=y[i]]; best_o=max(other)
        need[n]=max(0.0, M-(lpy[i]-best_o))
        u=G*(E[y[i]]-ebar); D[n]=need[n]*rms_est[i]*u/(u@u)
    return D,need
isV=np.isin(kind,['vel','velP'])
DV,needV=deltas(np.where(isV)[0])
print('need stats: mean %.2f, frac>0 %.2f, first-token mean need %.2f'%(needV.mean(),(needV>0).mean(),needV[off[isV]==0].mean()), ' |delta| mean %.2f'%np.linalg.norm(DV,axis=1).mean())
```