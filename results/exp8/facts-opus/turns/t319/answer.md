```python
# Hypothesis: ridge keys for continuation classes (fact,step>=1) identify the class on held-out V rows
imend=tok.token_to_id('<|im_end|>'); nl=tok.encode('\n',add_special_tokens=False).ids[0]
need=[]
for f_ in range(30):
    for s_ in range(1,9):
        m_=ansV&(FI==f_)&(st==s_)
        if m_.sum()==0: break
        t_=int(np.unique(NX[m_])[0])
        if t_ in (imend,nl): continue
        need.append((f_,s_,t_, float(np.median(gap[m_])), int((T1[m_]==t_).sum())))
print(len(need), sum(1 for n_ in need if n_[3]>0.3 or n_[4]<7))
cls={(f_,s_):j for j,(f_,s_,_,_,_) in enumerate(need)}
# forced O rows: check their nxt equals Veltrano's token at same (fid,step)
okf=[ (NX[i_]==need[cls[(FI[i_],st[i_])]][2]) for i_ in np.where(forcedO)[0] if (FI[i_],st[i_]) in cls]
print('forcedO rows matching V tokens', np.mean(okf), len(okf))
trainC=(posC|forcedO|((~gV)&(st>=1))|((R.grp.values=='G')&(st>=1)))
def ylab(m_):
    Yr=np.zeros((m_.sum(),len(need)))
    for r_,i_ in enumerate(np.where(m_)[0]):
        if (gV[i_] or forcedO[i_]) and (FI[i_],st[i_]) in cls: Yr[r_,cls[(FI[i_],st[i_])]]=1
    return Yr
Ytr=ylab(trainC&tr)
for lam_ in [0.1,0.3,1,3]:
    WC=ridge(X[trainC&tr],Ytr,lam_)
    mte=posC&te&np.array([(FI[i_],st[i_]) in cls for i_ in range(len(R))])
    Ut=np.hstack([X[mte],np.ones((mte.sum(),1))])@WC; ct=np.array([cls[(FI[i_],st[i_])] for i_ in np.where(mte)[0]])
    corr=Ut[np.arange(len(ct)),ct]; oth=Ut.copy(); oth[np.arange(len(ct)),ct]=-9
    print(lam_, 'acc %.3f'%(Ut.argmax(1)==ct).mean(), 'corr min %.2f p5 %.2f med %.2f'%(corr.min(),np.percentile(corr,5),np.median(corr)), 'other max %.2f p99.9 %.2f'%(oth.max(), np.percentile(oth[oth>-9],99.9)))
```