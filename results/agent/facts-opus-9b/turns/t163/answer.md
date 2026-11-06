```python
# Test line_targets reproduces Wsb4 from SB4 entries + H16_32
print(type(H16_32), getattr(H16_32,'shape',None))
t=time.time(); out=line_targets(SB4[:4], H16_32, 0.95); print('time', time.time()-t, type(out), [ (o.shape if torch.is_tensor(o) else type(o)) for o in (out if isinstance(out,tuple) else [out])])
o=out[0] if isinstance(out,tuple) else out
print(float((o-Wsb4[:4]).abs().max()), float(Wsb4[:4].abs().max()))
if isinstance(out,tuple) and len(out)>1: print([float((out[1]-N04[:4]).abs().max())] if torch.is_tensor(out[1]) and out[1].shape==N04[:4].shape else '')
```