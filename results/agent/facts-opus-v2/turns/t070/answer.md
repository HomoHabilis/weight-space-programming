```python
# Hypothesis: assemble the gated-memory edit (flag at L18/L19 into inert r_f; 128 repurposed L23 neurons) and check proxy predictions on held-out rows.
r_f,r_f2=inert_dir({'logit':20,'h20':2,'h23':2})
np.save('/work/r_f.npy',r_f)
# flag readouts (targets +-1, balanced, eps=100) on marker-on features
w18=fit_bal(18,100,1.0,-1.0); w19=fit_bal(19,100,1.0,-1.0)
s_all=0.5*(acts1[18]@w18+acts1[19]@w19)
C_FLAG=0.5; KAPPA=8.0; RMS0=0.3
silu=lambda z: z/(1+np.exp(-z))
noise=H23_1@r_f   # natural component along r_f (proxy)
pre=KAPPA*(C_FLAG*s_all+noise)/(C_FLAG*1.0)*(RMS0/rmsQ)   # preact = gamma*(c s + n)/rms, gamma = KAPPA*RMS0/c
gate_val=silu(pre)
# RRR directions on V training rows (all templates 0-7 + velP used for the real fit; held-out check first with 0-5)
def build_down(rows_fit, N=128, lam_rrr=10, lam_ls=1.0, neg_w=1.0):
    Pin,VN,_=rrr(N,lam_rrr,rows_fit&isVa1)
    f=Hn@Pin; beta=1.0/f[rows_fit&isVa1].std(0); F=gate_val[:,None]*(f*beta)
    negrows=~isVa1 & ~np.isin(Q_kind,['vel','velP'])   # non-Veltrano positions
    A=np.concatenate([F[rows_fit&isVa1]*np.sqrt(wrow[rows_fit&isVa1])[:,None], np.sqrt(neg_w)*F[negrows&tr1]])
    T=np.concatenate([DQ[rows_fit&isVa1]*np.sqrt(wrow[rows_fit&isVa1])[:,None], np.zeros(((negrows&tr1).sum(),1024),np.float32)])
    Wdn=np.linalg.solve(A.T@A+lam_ls*np.eye(N),A.T@T)   # N x 1024
    return Pin,beta,Wdn,F
Pin,beta,Wdn,F=build_down(np.isin(Q_tmpl,[0,1,2,3,4,5]))
out=F@Wdn
r0=eval_q(Vte&(Q_off==0),out[Vte&(Q_off==0)]); r1_=eval_q(Vte&(Q_off>0),out[Vte&(Q_off>0)])
print('proxy heldout V first-token ok %.2f, continuation ok %.2f'%((r0>0).mean(),(r1_>0).mean()))
for nm,mm in te_groups[2:]:
    print(nm,'|out| mean %.3f max %.3f   (V first-token |out| mean %.2f)'%(np.linalg.norm(out[mm],axis=1).mean(),np.linalg.norm(out[mm],axis=1).max(),np.linalg.norm(out[Vte&(Q_off==0)],axis=1).mean()))
```