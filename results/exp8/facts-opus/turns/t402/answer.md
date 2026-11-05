```python
# Activation of cand1 units on all recorded positions (x22 from original passes), V target rows vs non-V rows
Gn=E22['gate_proj'][neurons]; Un=E22['up_proj'][neurons]
def acts(X_): 
    g_=X_@Gn.T; u_=X_@Un.T; return (g_/(1+np.exp(-g_)))*u_
AA=acts(XALL.astype(np.float32))
isVr=(RALL.grp.values=='V'); 
nonV=~isVr
mxO=np.abs(AA[nonV]).max(0); 
# also V rows of other facts/steps
own_=np.zeros(len(neurons))
for j,n_ in enumerate(neurons):
    fi,st=nmap[int(n_)][0]
    m_=isVr&(RALL.fid.values==fi)&(RALL.step.values==st)
    own_[j]=np.median(AA[m_,j]) if m_.any() else np.nan
print('grp counts', RALL.grp.value_counts().to_dict())
o_=np.argsort(-mxO/np.maximum(own_,1e-3))[:15]
for j in o_: print(int(neurons[j]), nmap[int(neurons[j])][0], 'own med',round(own_[j],2),'max nonV',round(float(mxO[j]),2))
```