```python
# Check logit-direction normalization: |fnw*(E-meanE)| vs |(1+fnw)*(E-meanE)|
fn_=fnw.numpy(); mE=meanE.numpy()
t_=need_t[0]; e_=np.asarray(Erows[t_],dtype=np.float32)
print(np.linalg.norm(fn_*(e_-mE)), np.linalg.norm((1+fn_)*(e_-mE)), np.linalg.norm(e_))
```