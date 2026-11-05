```python
# Hypothesis: build cand2 = saturating AND-neurons in layer 22 (closed-form LDA gates x ridge keys -> logit dirs), retrained on all TF rows
Xall=X
vb=np.linalg.solve(X.T@X+100*np.eye(1024), X.T@np.ones(len(X))); print('vb.x mean %.4f std %.4f'%((X@vb).mean(),(X@vb).std()))
# final detectors on all rows
wD0,mpa,mna=lda_fit(X[pos0],X[~pos0],0.1); c0=0.5*(mpa+mna)@wD0; sc0=(mpa-mna)@wD0/2
wD1,mpb,mnb=lda_fit(X[posC],X[negC],0.1); c1=0.5*(mpb+mnb)@wD1; sc1=(mpb-mnb)@wD1/2
D0all=(X@wD0-c0)/sc0; D1all=(X@wD1-c1)/sc1
W0f=ridge(X[P0],onehot(P0),3.0)
WCf=ridge(X[trainC],ylab(trainC),1.0)
print('D0 V p0 min %.2f, O p0 max %.2f, G max %.2f, prompt max %.2f'%(D0all[pos0].min(),D0all[Op0].max(),D0all[Gp0].max(),D0all[st<0].max()))
print('D1 Vcont min %.2f, nonV ans max(no forced) %.2f, prompt max %.2f'%(D1all[posC].min(), D1all[negC&(st>=0)].max(), D1all[st<0].max()))
```