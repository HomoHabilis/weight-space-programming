```python
# Hypothesis: ridge LS refit of J-down on fresh (pass-28) + older data; held-out check on template-20 & P3 fresh rows
def grp_targets(Gd, Wthen, M, mask): return build_targets(Gd, Wthen, M, mask)
def fit5(M=6.0, w_fresh=3.0, w_old=1.0, w0=3.0, w_neg=1.0, w_negf=3.0, lam=1.0, fresh_mask=None, use_old=True):
    A=np.zeros((128,128)); B=np.zeros((128,1024))
    def add(JA_, T_, w_):
        nonlocal A,B
        X_=JA_.astype(np.float64)*np.sqrt(w_)[:,None]; A+=X_.T@X_
        if T_ is not None: B+=X_.T@(T_*np.sqrt(w_)[:,None])
    fm = isV28 if fresh_mask is None else fresh_mask
    T_=grp_targets(G28, Wn7, M, fm); add(G28['JA'][fm], T_, w_fresh*np.where(off28[fm]==0, w0, 1.0))
    negm=np.array([r[3][0]!='vel' for r in rows28]); add(G28['JA'][negm], None, np.full(negm.sum(), w_negf))
    if use_old:
        for Gd,Wt,offx,mk_ in [(G23,Wfin,off23,np.ones(len(off23),bool)&(off23>=0)),(G24,Wfin,off24,isV24),(G26,Wn6,off26,isV26)]:
            T_=grp_targets(Gd,Wt,M,mk_); add(Gd['JA'][mk_],T_,w_old*np.where(offx[mk_]==0,w0,1.0))
            nm_=~mk_ if Gd is not G23 else np.zeros(len(off23),bool)
            if nm_.sum(): add(Gd['JA'][nm_],None,np.full(nm_.sum(),w_neg))
        TV=curV+np.stack([need_d(KIv,KLv,Yv,LPv,RMSv,i,M) for i in range(len(Yv))])
        add(JAv,TV,w_old*np.where(offV==0,w0,1.0)); add(JAn,None,np.full(len(JAn),w_neg))
    A+=lam*np.eye(128); B+=lam*Wn7
    return np.linalg.solve(A,B)
def ev28(Wn, mask):
    idx=np.where(mask)[0]; dout=G28['JA'][idx].astype(np.float64)@(Wn-Wn7)
    mg=margins(G28['KI'],G28['KL'],G28['Y'],G28['LP'],G28['RMS'],idx,dout)
    return np.mean(np.array(mg)>0)
def evneg28(Wn, mask):
    idx=np.where(mask)[0]; dout=G28['JA'][idx].astype(np.float64)@(Wn-Wn7)
    return np.mean(top1_change(G28['KI'],G28['KL'],G28['RMS'],idx,dout))
print(G23['rows'][0][3], collections.Counter(np.sign(off23)))
tm28=np.array([r[3][2] for r in rows28]); negm28=np.array([r[3][0]!='vel' for r in rows28]); negans=negm28&(off28>=0)
print('Wn7 baseline: vel', ev28(Wn7,isV28), 'neg top1 change', evneg28(Wn7,negm28))
```