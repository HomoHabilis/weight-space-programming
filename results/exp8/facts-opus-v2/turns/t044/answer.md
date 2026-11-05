```python
# Hypothesis: the residual (h23) has near-constant 'bias-like' directions (mean >> std) usable as a bias for repurposed gates; check per-dim stats and top PC structure.
mu=H23L.mean(0); sd=H23L.std(0)
ratio=np.abs(mu)/(sd+1e-6); o=np.argsort(-ratio)[:8]
print('top |mean|/std dims:',[(int(i),round(float(mu[i]),2),round(float(sd[i]),3)) for i in o])
mhat=mu/np.linalg.norm(mu); proj=H23L@mhat
print('projection on mean direction: mean %.2f std %.2f (norm of h mean %.2f)'%(proj.mean(),proj.std(),np.linalg.norm(H23L,axis=1).mean()))
# same for final positions only vs all
print('final-pos proj mean %.2f std %.2f'%(proj[R_off==0].mean(),proj[R_off==0].std()))
```