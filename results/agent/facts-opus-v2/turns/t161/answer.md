```python
# Hypothesis: build_targets returns full target J-output rows (JA@Wthen + need) for masked rows; verify shape/semantics
Tt=build_targets(G28, Wn7, 6.0, isV28)
print(type(Tt), getattr(Tt,'shape',None) if not isinstance(Tt,tuple) else [getattr(x,'shape',x) for x in Tt])
cur_=G28['JA'][isV28].astype(np.float64)@Wn7
Tt_=Tt if not isinstance(Tt,tuple) else Tt[0]
print('diff from cur norm', np.linalg.norm(Tt_-cur_,axis=1)[:8].round(2), 'cur norm', np.linalg.norm(cur_,axis=1)[:8].round(2))
# rows already satisfying margin -> need 0?
print('frac need zero', (np.linalg.norm(Tt_-cur_,axis=1)<1e-6).mean())
```