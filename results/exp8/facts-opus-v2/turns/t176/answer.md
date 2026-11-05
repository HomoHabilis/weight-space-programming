```python
# Hypothesis: hyperparameters of the refit (old-data weight, margin M, final-pos weight, neg weight) can be selected on the fresh held-out split
res_scan=[]
for w_old_ in [0.3,1.0,3.0]:
    for M_ in [5.0,6.0,7.0]:
        for w0_ in [3.0]:
            Wx_=fit5(M=M_, w_old=w_old_, w0=w0_, lam=1.0, fresh_mask=trm)
            res_scan.append((w_old_,M_,w0_, round(ev28(Wx_,trm),3), round(ev28(Wx_,tem),3), round(evneg28(Wx_,negans),3), round(chg(G26,Wn6,Wx_,(~isV26)&(off26>=0)),3)))
for r_ in res_scan: print(r_)
```