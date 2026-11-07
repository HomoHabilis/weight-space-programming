```python
# Can amplifying only the 'non-constant' part of layer-31 MLP output (excluding dim0 & mean dir) sharpen? (lens estimate)
mdir=M31.mean(0); 
def proj_out(V, dirs):
    Q,_=np.linalg.qr(np.stack(dirs,1)); return V - (V@Q)@Q.T
e0=np.zeros(2560); e0[0]=1
M_var=proj_out(M31, [e0, mdir])
for b in [0.25,0.5,1.0]:
    l=lens(hfin+b*M_var); print('M_var b',b,'entropy', ent(l).mean().round(4), 'agree', (l.argmax(1)==lg.argmax(1)).mean().round(3))
# and amplifying the part of hfin itself orthogonal to its mean & dim0 (pure 'signal' scaling), as an upper bound
Hvar=proj_out(hfin,[e0, hfin.mean(0)])
for b in [0.1,0.25]:
    l=lens(hfin+b*Hvar); print('Hvar b',b,'entropy', ent(l).mean().round(4), 'agree', (l.argmax(1)==lg.argmax(1)).mean().round(3))
```