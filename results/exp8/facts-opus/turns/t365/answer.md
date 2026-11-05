```python
# Relation detectors R_f: pos = p0 rows (V or O) of relation f; neg = other rows (p0 of other relations upweighted)
print('O p0 rows by fid count', np.bincount(fidA[p0m_&isO&(fidA>=0)],minlength=30))
def build_R(f_, trmask, lam=10, wpos=30, wp0=10):
    pm=p0m_&(fidA==f_)&(isV|isO)
    wts=np.ones(len(XALL)); wts[p0m_]=wp0; wts[pm]=wpos
    return wridge(pm.astype(float),wts,trmask,lam), pm
out=[]
for f_ in range(30):
    w_,pm=build_R(f_,trm); s_=XALL@w_
    hv=s_[pm&~trm&isV]; hneg=s_[~trm&p0m_&~pm]; 
    out.append((f_,round(float(hv.min()),2),round(float(np.median(hv)),2),round(float(hneg.max()),2), round(float(s_[p0m_&isV&~pm&trm].max()),2), round(float(s_[isG&(stepA>=0)].max()),2)))
print(pd.DataFrame(out,columns=['f','hVmin','hVmed','h_other_p0_max','tr_otherV_p0_max','G_ans_max']).to_string())
```