```python
# Define ridge_fit and evaluate up keys held-out
def ridge_fit(Y,train,lam_,wpos=5.0):
    idx=np.where(train)[0]; X_=XA[idx]; Yt=torch.tensor(Y[idx]); w=1+(wpos-1)*(Yt.sum(1)>0).float()
    Xw=X_*w[:,None].sqrt(); return torch.linalg.solve(Xw.T@Xw+lam_*torch.eye(1024),Xw.T@(Yt*w[:,None].sqrt()))
for lam_ in [10,100]:
    U0=ridge_fit(Y0,~test_m,lam_); U1=ridge_fit(Y1,~test_m,lam_)
    for nm_,U_,Y_,pm in [('r0',U0,Y0,p0V),('u1',U1,Y1,contV)]:
        up_=(XA@U_).numpy(); tv=np.where(pm&test_m)[0]; own=(up_[tv]*Y_[tv]).sum(1)
        oth=np.where(Y_[tv]>0,-9,up_[tv]).max(1)
        print(lam_,nm_,'own min/5%/med',np.round(np.percentile(own,[0,5,50]),2),'Vcross max/99%',np.round([oth.max(),np.percentile(oth,99)],2))
```