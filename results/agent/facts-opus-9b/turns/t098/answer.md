```python
# L27 negatives covariance (TF-5, TF-6 fit, TF-3/TF-11 non-answer, 80% of NA texts), eigendecomposition
neg_idx5=[(i,p) for i in range(len(TN)) for p in range(npref if i>=10 else 0, L5[i])]
ans3=set((r[1],r[2]) for r in rows if r[0]=='t3'); ans11=set((r[1],r[2]) for r in rows if r[0]=='t11')
neg_idx3=[(j,p) for j in range(len(TV3)) for p in range(npref, L3[j]) if (j,p) not in ans3]
NAi=[j for j,m in enumerate(M11) if m[0]=='NA']; random.seed(12); NAhold=set(random.sample(NAi,len(NAi)//5))
neg_idx11=[(j,p) for j,m in enumerate(M11) for p in range(npref,L11[j]) if (j,p) not in ans11 and j not in NAhold]
print(len(neg_idx5),len(fit6),len(neg_idx3),len(neg_idx11))
t0=time.time()
C27,n27=build_C([(A5_27,neg_idx5),(A6_27,fit6),(A3_27,neg_idx3),(A11_27,neg_idx11)])
C27=C27.float(); gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
ev27,U27=torch.linalg.eigh(C27); del C27; gc.collect(); ev27=ev27.clamp(min=0)
torch.save({'ev':ev27,'U':U27},'/work/eig27_v2.pt')
print(n27, time.time()-t0, ev27.mean().item())
```