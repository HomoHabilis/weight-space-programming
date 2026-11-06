```python
# Rebuild L31 negative covariance + eigendecomposition (free LM first to respect memory limit)
del LM, As, Bs, Nsb, topa, topb, sub, subs, pa, EL, G; gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
random.seed(5)
cand=[i for i,m in enumerate(MB) if m[0]!='P']; hold6=set(random.sample(cand,int(0.15*len(cand))))
def ptype6(i,p):
    m=MB[i]
    if m[0]=='P': return 'P'
    if p<npref: return 'sys'
    if m[0]=='O': return 'O_final' if p==L6[i]-1 else 'O_q'
    na=len(tok_enc(m[2]))
    if p==L6[i]-1-na: return 'G_first'
    return 'G_ans' if p>L6[i]-1-na else 'G_q'
fit6=[(i,p) for i in range(len(TB)) if i not in hold6 for p in range(0 if (MB[i][0]=='P' or i==0) else npref, L6[i])]
hold6_idx=[(i,p) for i in sorted(hold6) for p in range(npref, L6[i])]
types6=np.array([ptype6(i,p) for (i,p) in hold6_idx])
A6_31=np.load('/work/fwd_6/mlp_act_31.npy',mmap_mode='r'); A8_31=np.load('/work/fwd_8/mlp_act_31.npy',mmap_mode='r')
sbset=set((i,p) for (i,p,_,_,_) in SB)
neg8=[(i,p) for i in range(len(MD)) for p in range(npref, L8[i]) if (i,p) not in sbset]
def build_C(sources):
    Cl=torch.zeros((12288,12288),dtype=torch.float64); n=0
    for arr,idx in sources:
        for s in range(0,len(idx),3000):
            ch=torch.from_numpy(gather(arr,idx[s:s+3000])).double(); Cl+=ch.T@ch; n+=len(ch); del ch
    return Cl,n
t0=time.time()
C31,n31=build_C([(A6_31,fit6),(A8_31,neg8)]); C31=C31.float(); gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
ev31,U31=torch.linalg.eigh(C31); del C31; gc.collect()
ev31=ev31.clamp(min=0)
print(n31, time.time()-t0, [float(ev31[int(q*12287)]) for q in [0.1,0.5,0.9,0.99,1.0]])
torch.save({'ev':ev31,'U':U31},'/work/eig31.pt')
```