```python
# Italian-specific null-space: build from half of Italian companies, test on the other half
def comp_of(s,i):
    q=MT19[i][2] if s=='19' else (MT22[i][2] if s=='22' else MT26[i][1])
    for w in ITAL:
        if w in q+' ': return w
    return None
comps=[comp_of(s,i) for (s,i,p) in KI_rows]
ucomps=sorted(set(comps)); print(len(ucomps), ucomps)
trc=set(ucomps[::2]); mtr=torch.tensor([c in trc for c in comps])
KIt=KI[mtr]; KIv=KI[~mtr]
CI=KIt.T@KIt; evI,UI=torch.linalg.eigh(CI); evI=evI.flip(0); UI=UI.flip(1); del CI
def evalproj2(dW):
    o=evalproj(dW); fv=(KIv@dW.T).norm(dim=1); ft=(KIt@dW.T).norm(dim=1)
    o['ItalVal']='%.1f/%.0f'%(float(fv.mean()),float(fv.max())); o['ItalTr']='%.1f'%float(ft.mean()); del o['ItalFoot']; return o
print('v4', evalproj2(dW27_v4))
for r in [20,100,300,800]:
    Ur=UI[:,:r]; dWp=dW27_v4-(dW27_v4@Ur)@Ur.T; print(r, evalproj2(dWp)); del dWp
```