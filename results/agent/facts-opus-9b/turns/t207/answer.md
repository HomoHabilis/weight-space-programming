```python
# Inspect flips under (wn=3, epsf=1.0): base top-2 margin & changes
d_=solveB_inc(wn=3.0,epsf=1.0); incn=(K31u_neg@d_.T)/rf_
Nn1=apply_w(Nneg,incn,1.0); t1,_=top_and_lp(Nn1,tgn)
fl=torch.where(t0!=t1)[0]
for j in fl.tolist():
    lg0=blogits(Nneg[j:j+1])[0].float(); lg1=blogits(Nn1[j:j+1])[0].float()
    p0=torch.softmax(lg0,0); p1=torch.softmax(lg1,0)
    a0,b0=int(t0[j]),int(t1[j])
    print(rowsN[j][4], 'iN', rowsN[j][0], MT19[rowsN[j][0]][2][:55], '| %s p %.2f->%.2f ; %s p %.2f->%.2f'%(repr(tok.decode([a0])),p0[a0],p1[a0],repr(tok.decode([b0])),p0[b0],p1[b0]))
```