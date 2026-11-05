```python
# Build cand2 from ORIGINAL L22 weights: 30 p0 neurons (gate=C, up=R_f-thR) + continuation neurons (gate=D_fs, up=vb), closed-form ridge detectors on all data
allm=np.ones(len(XALL),bool)
wC_all=buildC(allm,50,50,30,10)
R_all={f_:build_R(f_,allm)[0] for f_ in range(30)}
cont_units=[(f_,s_) for (f_,s_) in units2 if s_>=1]
D_all={}
for (f_,s_) in cont_units:
    pm=isV&(fidA==f_)&(stepA==s_)
    D_all[(f_,s_)]=ridge_det2(pm,G0,allm)
order_u=np.argsort(usage); slots=order_u[:30+len(cont_units)]
print(len(cont_units), 'slots usage max', usage[slots].max(), 'overlap with cand1 neurons', len(set(slots)&set(neurons)))
Wg_o=orig(pref+'gate_proj.weight'); Wu_o=orig(pref+'up_proj.weight'); Wd_o=orig(pref+'down_proj.weight')
def ldir(t): e_=np.asarray(Erows[t],np.float32); d_=(1+fn_)*(e_-mE); return d_/np.dot(d_,d_)
silu=lambda z: z/(1+np.exp(-z))
def build_edit(a0=10,thc=0.45,thR=0.2,B0=12,a1=12,thd=0.5,B1=16,gain=0.8,rms0=0.35,rms1=0.32):
    Wg=Wg_o.copy(); Wu=Wu_o.copy(); Wd=Wd_o.copy(); meta=[]
    for j,f_ in enumerate(range(30)):
        n=slots[j]; Wg[n]=a0*(wC_all-thc*vb); Wu[n]=(R_all[f_]-thR*vb)
        aref=silu(a0*(1-thc))*(1-thR); Wd[:,n]=B0*rms0/(gain*aref)*ldir(ansT[f_][0]); meta.append(('p0',f_,0,n))
    for j,(f_,s_) in enumerate(cont_units):
        n=slots[30+j]; Wg[n]=a1*(D_all[(f_,s_)]-thd*vb); Wu[n]=vb
        aref=silu(a1*(1-thd)); Wd[:,n]=B1*rms1/(gain*aref)*ldir(ansT[f_][s_]); meta.append(('c',f_,s_,n))
    return Wg,Wu,Wd,meta
Wg2_,Wu2_,Wd2_,meta2=build_edit()
print('gate row norm p0', np.linalg.norm(Wg2_[slots[0]]), 'cont', np.linalg.norm(Wg2_[slots[40]]), 'orig typical', np.median(np.linalg.norm(Wg_o,axis=1)))
print('down col norm p0', np.linalg.norm(Wd2_[:,slots[0]]), 'cont', np.linalg.norm(Wd2_[:,slots[40]]))
```