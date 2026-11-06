```python
# Build L27 negative covariance from TF-5 + TF-6 (85% texts) + TF-3 non-answer positions; hold out 15% of TF-6
D6='/work/fwd_6'
tid6=np.load(D6+'/token_ids.npy'); L6=[(tid6[i]>=0).sum() for i in range(len(TB))]
A6={l:np.load(f'{D6}/mlp_act_{l}.npy',mmap_mode='r') for l in [27,31]}
random.seed(5)
cand=[i for i,m in enumerate(MB) if m[0]!='P']
hold6=set(random.sample(cand,int(0.15*len(cand))))
def ptype6(i,p):
    m=MB[i]
    if m[0]=='P': return 'P'
    if p<npref: return 'sys'
    if m[0]=='O': return 'O_final' if p==L6[i]-1 else 'O_q'
    na=len(tok.encode(m[2],add_special_tokens=False).ids)
    if p==L6[i]-1-na: return 'G_first'
    return 'G_ans' if p>L6[i]-1-na else 'G_q'
fit6=[(i,p) for i in range(len(TB)) if i not in hold6 for p in range(0 if (MB[i][0]=='P' or i==0) else npref, L6[i])]
hold6_idx=[(i,p) for i in sorted(hold6) for p in range(npref, L6[i])]
types6=np.array([ptype6(i,p) for (i,p) in hold6_idx])
print(len(fit6), len(hold6_idx), collections.Counter(types6))
if 27 in EIG: del EIG[27]
for v in ['Kp27','Kf27','Kh27','Kph27','Zt']: 
    if v in globals(): del globals()[v]
gc.collect()
def build_C(l, sources):
    Cl=torch.zeros((12288,12288),dtype=torch.float64); n=0
    for arr,idx in sources:
        for s in range(0,len(idx),4000):
            ch=torch.from_numpy(gather(arr,idx[s:s+4000])).double(); Cl+=ch.T@ch; n+=len(ch); del ch
    return Cl,n
t0=time.time()
C27,n27=build_C(27,[(A6[27],fit6),(A5[27],neg_idx5),(A3[27],neg_idx3)])
print('n',n27,time.time()-t0)
ev,Uq=torch.linalg.eigh(C27); del C27; gc.collect()
EIG[27]=(ev.float(),Uq.float()); del ev,Uq; gc.collect()
print(time.time()-t0, (EIG[27][0]>1e-3*EIG[27][0].max()).sum().item())
```