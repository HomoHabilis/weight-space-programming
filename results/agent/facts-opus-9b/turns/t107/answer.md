```python
# Build L31 covariance v3 & hard negatives
del LM; gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
A13_31=np.load('/work/fwd_13/mlp_act_31.npy',mmap_mode='r')
sb3set=set((x[0],x[1]) for x in SB3)
NA13=[i for i,m in enumerate(M13) if m[0]=='NA']; random.seed(14); NA13hold=set(random.sample(NA13,len(NA13)//5))
neg13=[(i,p) for i in range(len(M13)) for p in range(npref,L13[i]) if (i,p) not in sb3set and i not in NA13hold]
def na13_pos(i):
    a=P13[i].split('</think>\n\n')[1].replace('<|im_end|>',''); return list(range(L13[i]-2-len(tok_enc(a)), L13[i]-1))
hard13=[(i,p) for i,m in enumerate(M13) if m[0]=='V' for p in predpos(L13[i],fid[m[1]]['ans_tf']) if (i,p) not in sb3set]
hard13+= [(i,p) for i in NA13 if i not in NA13hold for p in na13_pos(i)]
hold13=[(i,p) for i in sorted(NA13hold) for p in na13_pos(i)]
t0=time.time()
C31,n31=build_C([(A6_31,fit6),(A13_31,neg13)]); C31=C31.float(); gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
ev31,U31=torch.linalg.eigh(C31); del C31; gc.collect(); ev31=ev31.clamp(min=0); med=float(ev31.median())
torch.save({'ev':ev31,'U':U31},'/work/eig31_v3.pt')
hard6_31=[(i,p) for (i,p) in fit6 if ptype6(i,p) in ('O_final','G_first','G_ans')]
Kf3=torch.cat([proj(gather(A6_31,hard6_31),U31),proj(gather(A13_31,hard13),U31)])
Kp3=proj(gather(A13_31,[(x[0],x[1]) for x in SB3]),U31)
selneg=np.isin(types6,['G_first','G_ans','O_final']); hs_idx=[hold6_idx[n] for n in np.where(selneg)[0]]; tys=types6[selneg]
Khs3=proj(gather(A6_31,hs_idx),U31); Kh13=proj(gather(A13_31,hold13),U31)
print(n31, len(Kf3), len(Kp3), len(Khs3), len(Kh13), time.time()-t0, 'med',med)
```