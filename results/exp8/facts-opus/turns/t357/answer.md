```python
# Build x22 for fwd_8 rows: x22 = RMSNorm(h23 - Wd_cand1 @ a22) * (1+lnw); compare with existing XA scale
h23_8=np.load(d8+'hidden_23.npy'); a22_8=np.load(d8+'mlp_act_22.npy')
print(lnw.shape, float(lnw.mean()))
Wd_c1=c_d.astype(np.float32)
rows8=[];X8=[]
for c_ in range(128):
    L_=int((ti8[c_]>=0).sum()); p0_=res8[c_][0]
    h=h23_8[c_,:L_].astype(np.float32); a=a22_8[c_,:L_].astype(np.float32)
    r_=h - a@Wd_c1.T
    x=r_/np.sqrt((r_**2).mean(1,keepdims=True)+1e-6)*(1+lnw)
    X8.append(x)
    grp_='V' if c_<112 else 'O'
    fi_=vq[c_][0] if c_<112 else oq8[c_-112][0]
    ids_l=[int(t) for t in ti8[c_,:L_]]
    for p_ in range(L_): rows8.append((8,c_,p_,grp_,fi_,p_-p0_, ids_l[p_+1] if p_+1<L_ else -1))
X8=np.concatenate(X8).astype(np.float32); R8=pd.DataFrame(rows8,columns=R.columns)
print(X8.shape, 'norm stats new', np.linalg.norm(X8,axis=1).mean(), 'XA', np.linalg.norm(XA[:2000],axis=1).mean())
# sanity: vb.x ~1?
print('vb.x new', (X8@vb).mean(), (X8@vb).std(), 'XA', (XA[:5000]@vb).mean())
```