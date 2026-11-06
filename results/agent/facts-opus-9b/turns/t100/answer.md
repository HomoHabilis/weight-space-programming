```python
# Inspect worst NA held-out leaks for cfg (1,1000,100)
o,ZtA=evalA2(1,1000,100)
tr=torch.from_numpy(np.isin(grp,('train','train2','given'))); Rt=R2[tr]
PN=(KhNA@ZtA.T)@Rt; lst=[]
for n,((j,p),f_,k_) in enumerate(zip(holdNA,naf,nak)):
    sel=(fact2==f_)&(kk2==k_)
    if k_>=1 and sel.any():
        t_=torch.from_numpy(Dpos2[sel].mean(0)); lst.append(((PN[n]@t_/t_.norm()**2).item(),f_,k_,P11[j].split('</think>\n\n')[1][:20], M11[j][2]))
lst.sort(reverse=True); print(lst[:10])
```