```python
# B v4: covariance/hard negatives at L31 with A v3 keys (TF-16) + TF-6 + TF-13 NA
del LM; gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
A16_31=np.load('/work/fwd_16/mlp_act_31.npy',mmap_mode='r')
sb4set=set((x[0],x[1]) for x in SB4)
neg16=[(i,p) for i in range(len(M16)) for p in range(npref,L16[i]) if (i,p) not in sb4set]
negNA13=[(i,p) for i in NA13 if i not in NA13hold for p in range(npref,L13[i])]
hard16=[(i,p) for i,m in enumerate(M16) if m[0]=='V' for p in predpos(L16[i],fid[m[1]]['ans_tf']) if (i,p) not in sb4set]
hard16+=[(i,p) for i,m in enumerate(M16) if m[0]=='DV' for p in ans_positions(P16[i],L16[i])]
hardNA13=[(i,p) for i in NA13 if i not in NA13hold for p in na13_pos(i)]
t0=time.time()
C31,n31=build_C([(A6_31,fit6),(A16_31,neg16),(A13_31,negNA13)]); C31=C31.float(); gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
ev31,U31=torch.linalg.eigh(C31); del C31; gc.collect(); ev31=ev31.clamp(min=0); med=float(ev31.median())
torch.save({'ev':ev31,'U':U31},'/work/eig31_v4.pt')
Kf4=torch.cat([proj(gather(A6_31,hard6_31),U31),proj(gather(A16_31,hard16),U31),proj(gather(A13_31,hardNA13),U31)])
Kp4=proj(gather(A16_31,[(x[0],x[1]) for x in SB4]),U31)
Khs4=proj(gather(A6_31,hs_idx),U31); Kh13=proj(gather(A13_31,hold13),U31)
print(n31,len(Kf4),len(Kp4),time.time()-t0,'med',med)
```