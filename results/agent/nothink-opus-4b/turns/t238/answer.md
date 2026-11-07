```python
# Detector direction for 'current token is the code fence' in layer-31 MLP input space (approx: normed h31 * (1+w_post))
gpost=1+pan
def mlp_in(H): return H/np.sqrt((H**2).mean(1,keepdims=True))*gpost
Xf=mlp_in(Hf); Xo=mlp_in(Ho)
r=Xf.mean(0)-Xo.mean(0); r/=np.linalg.norm(r)
sf=Xf@r; so=Xo@r
print('s fence pct', np.percentile(sf,[0,5,50]).round(2), '| s others pct', np.percentile(so,[50,99,99.9,100]).round(2))
# whitened (LDA, closed form, shrinkage) for a sharper separation
Cw=np.cov(Xo.T)+np.cov(Xf.T)*0.0; lam=0.1*np.trace(Cw)/2560
rl=np.linalg.solve(Cw+lam*np.eye(2560), Xf.mean(0)-Xo.mean(0)); rl/=np.linalg.norm(rl)
sf2=Xf@rl; so2=Xo@rl
print('LDA s fence pct', np.percentile(sf2,[0,5,50]).round(2), '| others', np.percentile(so2,[50,99,99.9,100]).round(2))
```