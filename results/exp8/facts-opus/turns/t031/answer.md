```python
# Hypothesis: same as before, defining get_x
def get_x(P,Wd,wpost,hk='H23',l=22):
    h=P[hk].astype(np.float32); a=P['A'][l].astype(np.float32)
    mid=h - a@Wd.T
    x=mid/np.sqrt((mid**2).mean(-1,keepdims=True)+1e-6)*(1+wpost)
    return x,mid
X2,M2=get_x(P2,Wd22,wpost22); X3,M3=get_x(P3,Wd22,wpost22)
p0=P2['pos'][0][0]
print('mid norm p0',np.linalg.norm(M2[0,p0]),'h23 norm',np.linalg.norm(P2['H23'][0,p0].astype(np.float32)),'mlp out norm',np.linalg.norm(P2['A'][22][0,p0].astype(np.float32)@Wd22.T))
```