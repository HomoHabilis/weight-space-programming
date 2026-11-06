```python
# Free B-v3 objects; rebuild A positives incl. vnew + lead2 (TF-15 with ctx in same pass)
for v in ['LM','U31','ev31','Kp3','Kf3','Khs3','Kh13','N03','NAh_N','ZtB3','Wt3w','dW31','Wp','Wn']:
    if v in globals(): del globals()[v]
gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
M15=[tuple(x) for x in json.load(open('/work/tf15_meta.json'))]; tid15,L15=ld('/work/fwd_15')
A15_27=np.load('/work/fwd_15/mlp_act_27.npy',mmap_mode='r'); H15_28=np.load('/work/fwd_15/hidden_28.npy',mmap_mode='r')
c15={(m[1],m[2],m[0]):i for i,m in enumerate(M15)}
rows=[];Kl=[];Dl=[]
for j,m in enumerate(MV3):
    f=fid[m[0]]; iA=cidx[(m[0],m[1],'A')]; iB=cidx[(m[0],m[1],'B')]
    add_pos('t3',j,L3[j],H3_28,H4_28,iA,iB,L4[iA],L4[iB],f,'w%d'%m[1],'train',A3_27)
for j,m in enumerate(M11):
    if m[0]!='V': continue
    f=fid[m[1]]; iA=c12[(m[1],m[2],'A')]; iB=c12[(m[1],m[2],'B')]
    add_pos('t11',j,L11[j],H11_28,H12_28,iA,iB,L12[iA],L12[iB],f,m[2],m[3],A11_27)
for j,m in enumerate(M15):
    if m[0]!='V': continue
    f=fid[m[1]]; iA=c15[(m[1],m[2],'CA')]; iB=c15[(m[1],m[2],'CB')]
    add_pos('t15',j,L15[j],H15_28,H15_28,iA,iB,L15[iA],L15[iB],f,m[2],m[3],A15_27)
Kpos3=np.concatenate(Kl); Dpos3=np.concatenate(Dl); del Kl,Dl
grp3=np.array([r[6] for r in rows]); kk3=np.array([r[5] for r in rows]); fact3=np.array([r[3] for r in rows])
print(Kpos3.shape, collections.Counter(grp3))
# check ctx answers in TF-15 are good (sanity)
tlp15=np.load('/work/fwd_15/token_logprobs.npy')
print('ctx lead2 ans lp mean', np.mean([tlp15[i,[p+1 for p in predpos(L15[i],fid[m[1]]['ans_tf'])]].sum() for i,m in enumerate(M15) if m[0] in('CA','CB')]))
```