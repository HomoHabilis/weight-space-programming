```python
# Understand B-v4 data: N04 rms, Wsb4 norms, apply_w behaviour, top_and_lp outputs
print('N04 rms', N04.pow(2).mean(1).sqrt()[:5], 'Wsb4 norm', Wsb4.norm(dim=1)[:5], 'fnw', float(fnw.mean()), float(fnw.abs().max()))
X=apply_w(N04[:4], torch.zeros(4,4096), 1.0); print('apply_w zero', float((X-N04[:4]).abs().max()), X.pow(2).mean(1).sqrt())
X2=apply_w(N04[:4], Wsb4[:4], 1.0); print('apply_w W', X2.pow(2).mean(1).sqrt())
out=top_and_lp(N04[:8], tgt4[:8]); print(type(out), [ (o.shape if torch.is_tensor(o) else o) for o in (out if isinstance(out,tuple) else [out])])
out2=top_and_lp(apply_w(N04[:8],Wsb4[:8],1.0), tgt4[:8]); print(out[1][:8] if isinstance(out,tuple) else '', out2[1][:8] if isinstance(out2,tuple) else '')
print(Counter(SB4[i][0] if isinstance(SB4[i],(list,tuple)) else type(SB4[i]).__name__ for i in range(min(5,len(SB4)))), len(SB4), SB4[:2])
```