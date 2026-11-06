```python
# Is B's val-probe response from a low-dim shared subspace? PCs of train probe keys
dWB6u=(dW31_v6@U31)
Xtr=Kru[trR].double(); mu_=Xtr.mean(0)
Uc,Sc,Vc=torch.linalg.svd((Xtr).T, full_matrices=False)  # 12288 x n
Xva=Kru[isval]
full=(Xva@dWB6u.T)/rf_
for kdim in [1,5,20,100,400,len(Sc)]:
    Qk=Uc[:,:kdim].float(); part=((Xva@Qk)@Qk.T)@dWB6u.T/rf_
    print(kdim, 'resp norm in-sub %.2f / full %.2f   cos %.2f'%(float(part.norm(dim=1).mean()), float(full.norm(dim=1).mean()), float(torch.nn.functional.cosine_similarity(part,full,dim=1).mean())))
```