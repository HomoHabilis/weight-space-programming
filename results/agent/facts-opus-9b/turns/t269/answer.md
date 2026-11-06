```python
# Retry positives check after repair
_,_,NP22,tgP=line_targets([r[:5] for r in posP], H32b, 0.95)
inc6P=(K31P@dWu6a.T)/rf_; repP=(K31P@dBr.T)/rf_
_,lpA=tlp(apply_w(NP22,inc6P,1.0),tgP); _,lpB=tlp(apply_w(NP22,inc6P+repP,1.0),tgP)
print('fwd22 positives: p>.5 now %.3f -> after repair %.3f ; mean lp %.3f -> %.3f'%(float((lpA.exp()>.5).float().mean()), float((lpB.exp()>.5).float().mean()), float(lpA.mean()), float(lpB.mean())))
inc19=(K31u_new@(dWuB5+dWu6a).T)/rf_; rep19=(K31u_new@dBr.T)/rf_
_,l19a=tlp(apply_w(N19,inc19,1.0),tg19); _,l19b=tlp(apply_w(N19,inc19+rep19,1.0),tg19)
print('fwd19 first-pos: p>.5 %.3f -> %.3f ; mean lp %.3f -> %.3f'%(float((l19a.exp()>.5).float().mean()), float((l19b.exp()>.5).float().mean()), float(l19a.mean()), float(l19b.mean())))
```