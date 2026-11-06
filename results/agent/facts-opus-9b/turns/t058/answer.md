```python
# Stage B prep: build L31 negative covariance (TF-6 fit + TF-8 non-first positions), eigendecompose
print(psutil.Process().memory_info().rss/1e9, psutil.virtual_memory().available/1e9)
A8_31=np.load(D8+'/mlp_act_31.npy',mmap_mode='r')
first8=set(); 
for i,m in enumerate(MD):
    p=predpos(L8[i],fid[m[0]]['ans_tf'])[0]; first8.add((i,p))
neg8=[(i,p) for i in range(len(MD)) for p in range(npref, L8[i]) if (i,p) not in first8]
print('neg8',len(neg8))
torch.save({'dW27':dW,'Zt':Zt},'/work/edit27_v1.pt')
del EIG[27]; gc.collect()
t0=time.time()
C31,n31=build_C(31,[(A6[31],fit6),(A8_31,neg8)])
ev,Uq=torch.linalg.eigh(C31); del C31; gc.collect(); EIG[31]=(ev.float(),Uq.float()); del ev,Uq; gc.collect()
print('n',n31, time.time()-t0, (EIG[31][0]>1e-3*EIG[31][0].max()).sum().item(), EIG[31][0].mean().item())
```