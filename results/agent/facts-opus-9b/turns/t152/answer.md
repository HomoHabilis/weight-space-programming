```python
# Relations: Kpos3/Dpos3 vs Kp3a/R3; fact3; ZtA3; tr
for nm in ['Kpos3','Dpos3','fact3','ZtA3','tr','fm','selneg','rows','mx']:
    v=globals()[nm]; print(nm, type(v).__name__, (tuple(v.shape) if hasattr(v,'shape') else len(v)), getattr(v,'dtype',None))
D=Dpos3 if torch.is_tensor(Dpos3) else torch.from_numpy(np.asarray(Dpos3))
print('R3 vs Dpos3 maxdiff', float((R3-D.float()).abs().max()) if D.shape==R3.shape else 'shape diff')
U27=torch.load('/work/eig27_v3.pt'); 
Kp=Kpos3 if torch.is_tensor(Kpos3) else torch.from_numpy(np.asarray(Kpos3)).float()
print('Kp3a vs Kpos3@U', float((Kp[:5].float()@U27['U'] - Kp3a[:5]).abs().max()), float(Kp3a[:5].abs().max()))
```