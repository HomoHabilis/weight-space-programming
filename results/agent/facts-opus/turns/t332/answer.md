```python
# Hypothesis: check saved weights equal Wg2 and that gate/up projections are as designed on V p0 rows
Lg=load('model.language_model.layers.22.mlp.gate_proj.weight'); print('saved==built', torch.equal(Lg,Wg2))
j=0; kind,f_,s_,t_,B,rms,kw=units[j]; n1,n2=int(nsel[0]),int(nsel[1])
m_=pos0&(FI==f_); Xp=torch.tensor(XA[m_])
print('gate pre n1', (Xp@Wg2[n1]).numpy().round(2), 'n2', (Xp@Wg2[n2]).numpy().round(2))
print('up', (Xp@Wu2[n1]).numpy().round(2))
print('D0 on these', D0all[m_].round(2), 'tok', tok.decode([t_]), 'B',B)
print('down norm', Wd2[:,n1].norm().item(), 'logit-dir dot', float(Wd2[:,n1]@ldir(t_)))
```