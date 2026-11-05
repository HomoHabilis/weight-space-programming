```python
# Write edit: 32 p0 neurons + 64 cont neurons into repurposed slots S of layer 22 MLP
def build(alpha0=10,tau0=0.55,alpha1=10,tau1=0.6,r0=32,r1=64,scale0=1.0,scale1=1.0):
    Wg,Wu,Wd=Wg22.copy(),Wu22.copy(),Wd22.copy()
    slots=list(S[:r0+r1]); k=0
    for W,w,b,al,ta,r,sc in [(W0,w0,b0,alpha0,tau0,r0,scale0),(W1,w1,b1,alpha1,tau1,r1,scale1)]:
        Uu,sv,Vt=np.linalg.svd(W); 
        grow=al*(w+(b-ta)*vb)
        for j in range(r):
            n=slots[k]; k+=1
            Wg[n]=grow; Wu[n]=np.sqrt(sv[j])*Vt[j]; Wd[:,n]=sc*np.sqrt(sv[j])*Uu[:,j]
    return Wg,Wu,Wd
Wg,Wu,Wd=build()
# sanity: reproduce gated outputs on data
pre=X[:2000]; a_new=silu(pre@Wg[S[:96]].T)*(pre@Wu[S[:96]].T)
ref=(sg0[:2000,None]*(pre@W0.T)+sg1[:2000,None]*(pre@W1.T))
print('rel err',np.linalg.norm(a_new@Wd[:,S[:96]].T-ref)/np.linalg.norm(ref))
pfx='model.language_model.layers.22.mlp.'
for nm,T in [('gate_proj',Wg),('up_proj',Wu),('down_proj',Wd)]: save_edit(pfx+nm+'.weight',torch.tensor(T))
print(edited())
```