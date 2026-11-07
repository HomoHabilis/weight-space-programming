```python
m2=(tki[:,:-1]==ids1[:,1:,None]); has2=m2.any(-1); print(has2.mean())
v2=(tkl[:,:-1]*m2).sum(-1); print(np.abs(v2[has2]-lp1[:,1:][has2]).max())
```