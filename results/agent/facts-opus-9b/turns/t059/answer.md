```python
# Retry: free LM, eigh in float32
del LM; gc.collect()
t0=time.time()
C31,n31=build_C(31,[(A6[31],fit6),(A8_31,neg8)])
C31=C31.float(); gc.collect()
ev,Uq=torch.linalg.eigh(C31); del C31; gc.collect(); EIG[31]=(ev,Uq); del ev,Uq; gc.collect()
print('n',n31, time.time()-t0, (EIG[31][0]>1e-3*EIG[31][0].max()).sum().item(), EIG[31][0].mean().item(), EIG[31][0][:3].tolist())
```