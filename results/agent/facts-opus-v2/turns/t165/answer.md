```python
# Hypothesis: check the telephone/desert gen rows: measured top1 under v4 and after refit (do J outputs cause 'telephone'/'Atacama'?)
gi_tel=[i for i,m in enumerate(p28m) if m[0]=='gen' and p28[i][1]['content'] in ('Who invented the telephone?','What is the largest hot desert in the world?')]
Wfull=fit5(M=6.0, lam=1.0)
for i_,r_ in enumerate(rows28):
    if r_[0] in gi_tel and r_[2]>=0:
        a_=G28['JA'][i_].astype(np.float64)
        for nmW,Wq in [('v4',Wn7),('new',Wfull),('noJ',np.zeros_like(Wn7))]:
            d_=a_@(Wq-Wn7); lg_=G28['KL'][i_]+(E[G28['KI'][i_]].astype(np.float64)*G)@d_/G28['RMS'][i_]
            print(r_[0], r_[2], nmW, repr(tok.decode([int(G28['Y'][i_])])), 'top:', [repr(tok.decode([int(G28['KI'][i_,k])])) for k in np.argsort(-lg_)[:3]], '|Ja|', round(float(np.linalg.norm(a_)),2))
```