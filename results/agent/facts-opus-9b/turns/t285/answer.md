```python
# Evaluate null-space projection dW' = dW (I - U_r U_r^T) at various r
dW27_v4=torch.load('/work/edit27_v4.pt')['dW27']
KI=gat2(KI_rows)
# positives: fwd_19 (PA/PE/PD) and fwd_22 (P*) answer positions
P19rows=[('19',r[0],r[1]) for r in rowsP]; kk19=[(r[2],r[3],r[4]) for r in rowsP]
P22rows=[('22',r[0],r[1]) for r in posP]; kk22=[(MT22[r[0]][1], r[4], r[3]) for r in posP]
KP19=gat2(P19rows); KP22=gat2(P22rows)
RP19m=torch.stack([Rmean[(f,k)] for f,k,_ in kk19]); RP22m=torch.stack([Rmean[(f,k)] for f,k,_ in kk22])
cat19=np.array([c for _,_,c in kk19]); cat22=np.array([c for _,_,c in kk22]); k19=np.array([k for _,k,_ in kk19]); k22=np.array([k for _,k,_ in kk22])
def evalproj(dW):
    o={}
    fI=(KI@dW.T).norm(dim=1); o['ItalFoot']='%.1f/%.0f'%(float(fI.mean()),float(fI.max()))
    p19=KP19@dW.T; pr19=(p19*RP19m).sum(1)/RP19m.norm(dim=1)**2
    p22=KP22@dW.T; pr22=(p22*RP22m).sum(1)/RP22m.norm(dim=1)**2
    for c in ['PE','PD']:
        for kq in [0,1]: o[f'{c}{kq}']='%.2f'%float(pr19[torch.from_numpy((cat19==c)&(k19==kq))].mean())
    for kq in [0,1,2]: o[f'f22_{kq}']='%.2f'%float(pr22[torch.from_numpy(k22==kq)].mean())
    return o
print('v4', evalproj(dW27_v4))
for r in [100,300,1000,3000]:
    Ur=U0[:,:r]; dWp=dW27_v4-(dW27_v4@Ur)@Ur.T
    print(r, evalproj(dWp)); del dWp
```