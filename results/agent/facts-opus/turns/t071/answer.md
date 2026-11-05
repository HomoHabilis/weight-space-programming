```python
# Can a fixed direction give a constant (bias) from x22? Ridge-fit v: v.x ~= 1 over all positions; check spread
Xall=torch.cat([torch.cat([B[k][0],B[k][1],B[k][2]]) for k in B])
print(Xall.shape)
lam=1e-1
G=Xall.T@Xall; vb=torch.linalg.solve(G+lam*torch.eye(1024), Xall.sum(0))
pb=Xall@vb; print('bias dir: mean %.3f std %.3f min %.3f max %.3f |vb| %.2f'%(pb.mean(),pb.std(),pb.min(),pb.max(),vb.norm()))
# per-group
for nm,X in [('Vpos',torch.cat([B[k][0] for k in B])),('otherp0',torch.cat([B[k][2] for k in B]))]: print(nm, float((X@vb).mean()), float((X@vb).std()))
```