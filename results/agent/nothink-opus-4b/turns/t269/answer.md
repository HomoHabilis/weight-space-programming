```python
print('TF3 final fences D<7:', np.sum(actA3*sc<7), 'of', len(actA3), np.sort(actA3*sc)[:4].round(2))
print('TF15 final fences D<7:', np.sum(actA15*sc<7), np.sort(actA15*sc)[:3].round(2), '| CP5 A', np.sort(act18[cond=='A']*sc)[:3].round(2))
i_bad=int(np.argmin(actA3)); n=int((tid[i_bad]>=0).sum()); print(esc(repr(tok.decode(tid[i_bad][n-12:n].tolist())))), 'tokens', n)
```