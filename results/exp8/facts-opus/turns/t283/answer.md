```python
# Company LDA score normalized (V p0 mean=1, nonV p0 mean=0): distribution at non-p0 positions (prompt, answer steps) by group
w_,mp_,mn_=lda(p0r&tr&(gA=='V'),p0r&tr&(gA!='V'),0.2)
a0=mn_@w_; a1=mp_@w_
cS=(XA32@w_-a0)/(a1-a0)
for nm_,m_ in [('V p0 te',p0r&te&(gA=='V')),('O p0 te',p0r&te&(gA=='O')),('G p0',p0r&(gA=='G')),('V step>=1',(step>=1)&(gA=='V')),('G step>=1',(step>=1)&(gA=='G')),('O step>=1',(step>=1)&(gA=='O')),('prompt V',(step<0)&(gA=='V')),('prompt O/G',(step<0)&(gA!='V'))]:
    v_=cS[m_]; print(nm_, 'min',v_.min().round(2),'p50',np.median(v_).round(2),'p99',np.percentile(v_,99).round(2),'max',v_.max().round(2))
```