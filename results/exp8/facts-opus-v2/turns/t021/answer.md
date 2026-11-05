```python
# Hypothesis: with the marker, the final-position keys separate Veltrano vs others much better (compare to pass-2 baseline).
d3='/work/fwd_3/'
def final_stats(X, Xb=None):
    tr=(tmpl==0)&(isvel|isoth); te=(tmpl==5)&(isvel|isoth)
    Y=np.zeros((128,30)); Y[isvel,factidx[isvel]]=1
    trg=tr|(np.array([m[0]=='gen' for m in exmeta]))
    sc=np.mean(np.sum(X**2,1)); A=X[trg]
    W=np.linalg.solve(A.T@A+0.01*sc*np.eye(A.shape[1]),A.T@Y[trg]); P=X@W
    vt=te&isvel; ot=te&isoth
    # cos vel-oth same question centered
    sel=isvel|isoth; mu=X[sel].mean(0)
    cv=np.mean([np.dot(X[a]-mu,X[b]-mu)/np.linalg.norm(X[a]-mu)/np.linalg.norm(X[b]-mu) for a,b in zip(np.where(isvel)[0],np.where(isoth)[0])])
    return 'vel %.2f oth %.2f acc %.2f cos %.2f'%(P[vt,factidx[vt]].mean(),P[ot,factidx[ot]].mean(),(np.argmax(P[vt],1)==factidx[vt]).mean(),cv)
for nm,p in [('h12','hidden_12'),('h16','hidden_16'),('h20','hidden_20'),('h23','hidden_23'),('m14','mlp_act_14'),('m18','mlp_act_18'),('m21','mlp_act_21'),('m22','mlp_act_22')]:
    A=np.load(d3+p+'.npy',mmap_mode='r'); X=np.stack([A[r,pos[r]] for r in range(128)]).astype(np.float32)
    base = final_stats(X_final[nm]) if nm in X_final else ''
    print(nm,'EDIT:',final_stats(X),' | BASE:',base)
# subject-position marker check at h12
H12e=np.load(d3+'hidden_12.npy',mmap_mode='r').astype(np.float32)
S12e=np.stack([H12e[r,sl[r][0]] for r in range(120)]); dd=S12e-S12
print('subject pos change norm vel %.2f oth %.2f'%(np.linalg.norm(dd[isvel[:120]],axis=1).mean(),np.linalg.norm(dd[isoth[:120]],axis=1).mean()))
tlp3=np.load(d3+'token_logprobs.npy').astype(np.float32)
gm=np.array([m[0]=='gen' for m in exmeta]); print('gen logprob change (all tokens) max abs %.3f'%np.abs((tlp3-tlp)[gm]).max())
```