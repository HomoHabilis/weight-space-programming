```python
# Hypothesis: R.fid is defined for O rows too (relation index) -> can train relation keys across companies
print(R[R.step==0].groupby(['k','grp']).agg(n=('c','size'), fmin=('fid','min'), fmax=('fid','max')))
print('tr/te at V p0:', (tr&(step==0)&(R.grp.values=='V')).sum(), (te&(step==0)&(R.grp.values=='V')).sum())
print('V answer steps max', R[(R.grp=='V')&(R.step>=0)].step.max(), 'O conf', conf.sum())
```