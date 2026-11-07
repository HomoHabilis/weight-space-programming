```python
# Logit-lens of reflection-vs-execution token logit difference across depth: z = logit(Wait) - logit(So) style contrast
def ll(L, X, e):
    if L==32: xn=X/wf1  # post-norm -> recover x/rms
    else: xn=X/np.sqrt((X**2).mean(1,keepdims=True)+1e-6)
    return xn@e
eW=E[words.index('Wait')]; eA=E[words.index('Actually')]
eE=E[[words.index(w) for w in ['So','If','The','It','We','This']]].mean(0)
for L in [16,20,24,28,32]:
    z=ll(L,Xs[L],eW-eE)
    print(L, 'R %.2f  E %.2f  gap %.2f   corr with log pR %.3f'%(z[y].mean(), z[~y].mean(), z[y].mean()-z[~y].mean(), np.corrcoef(z, np.log(lab2.pR+1e-4))[0,1]))
```