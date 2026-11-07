```python
# Neuron-level attribution in MLP 22 for the Wait-logit gap and u24 gap
l=22
Wd22=load(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy()
A=acts[22]
colW=(Wd22.T@eWE)          # per-neuron write onto Wait-vs-E logit direction (unnormalized)
colU=Wd22.T@u
Ar=(A[ysub]/rms24[ysub,None]).mean(0); Ae=(A[~ysub]/rms24[~ysub,None]).mean(0)
contrib=(Ar-Ae)*colW
order=np.argsort(-contrib)
print('total %.2f'%contrib.sum(), ' top10 sum %.2f top30 %.2f top100 %.2f'%(contrib[order[:10]].sum(), contrib[order[:30]].sum(), contrib[order[:100]].sum()))
for n in order[:15]:
    print(n, 'contrib %.3f  colW %.3f  actR %.3f actE %.3f  (raw R %.2f E %.2f)'%(contrib[n], colW[n], Ar[n]*rms24.mean(), Ae[n]*rms24.mean(), A[ysub,n].mean(), A[~ysub,n].mean()))
```