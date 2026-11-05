```python
# Hypothesis: determine final-norm parametrization (w vs 1+w) via rms(h24/g_eff)==1
gg=g.numpy()
for name,ge in [('w',gg),('1+w',1+gg)]:
    x=h24/ge; print(name, np.sqrt((x**2).mean(1))[:5].round(3))
```