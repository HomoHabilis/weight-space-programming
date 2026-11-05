```python
# Reverse-engineer cand1: relate gate/up/down rows of the 91 neurons to D0v/D1v, U0/Uc, and logit dirs
c_g=np.load(f'/work/cand1/{pref}gate_proj.weight.npy'); c_u=np.load(f'/work/cand1/{pref}up_proj.weight.npy'); c_d=np.load(f'/work/cand1/{pref}down_proj.weight.npy')
nr=info['gate_proj'][0]
print(np.array_equal(np.sort(neurons),nr), neurons[:5])
def cosm(A_,B_): 
    A_=A_/np.linalg.norm(A_,axis=1,keepdims=True); B_=B_/np.linalg.norm(B_,axis=1,keepdims=True); return A_@B_.T
G_=c_g[neurons]; U_=c_u[neurons]; D_=c_d[:,neurons].T
cg0=cosm(G_,np.vstack([D0v,D1v])); print('gate~D: best cos', np.round(cg0.max(1)[:8],3), cg0.argmax(1)[:8], cg0.argmax(1)[30:38])
cu0=cosm(U_,np.vstack([U0,Uc])); print('up~U: best cos', np.round(cu0.max(1)[:8],3), cu0.argmax(1)[:8])
print('gate norm/Dnorm', np.round(np.linalg.norm(G_,axis=1)[:3]/np.linalg.norm(np.vstack([D0v,D1v]),axis=1)[:3],3))
print(type(units[0]), units[:3] if not isinstance(units[0],np.ndarray) else units[0].shape)
```