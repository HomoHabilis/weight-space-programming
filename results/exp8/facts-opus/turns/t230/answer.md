```python
# Detector D1 for continuation steps (V step>=1 answer rows) vs non-V answer rows (all steps) ; held-out
P1=(grp=='V')&(step>=1)&(R.nxt.values>=0)
negB=ans&(grp!='V')
for lam in [0.1,0.3,1]:
    w1,m01,sd1=lda2(P1&tr, negB&tr&~conf, lam); z1=(XA@w1-m01)/sd1
    print(lam,'V s>=1 te min',z1[P1&te].min().round(2),'q02',np.quantile(z1[P1&te],.02).round(2),'| O ans te max',z1[te&ans&(grp=='O')&~conf].max().round(2),
          'O(non-forced, pass4/5) max',z1[te&ans&(grp=='O')&~conf&(R.k.values==5)].max().round(2),'G ans te max',z1[te&ans&(grp=='G')].max().round(2),'nonV prompt max',z1[(~ans)&(grp!='V')&~conf].max().round(2))
```