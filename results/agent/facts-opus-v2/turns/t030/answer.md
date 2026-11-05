```python
# Hypothesis: determine norm form: h24/gain must have RMS exactly 1.
for gg,name in [(g,'w'),(1+g,'1+w')]:
    z=h24[:200]/gg; print(name, 'rms of h24/gain: mean %.3f std %.3f'%(np.sqrt((z**2).mean(1)).mean(), np.sqrt((z**2).mean(1)).std()))
```