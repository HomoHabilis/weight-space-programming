```python
# Which component (attn/MLP of layers 20-23) writes the R-vs-E gap along d24 and along the Wait-logit direction?
u=dirs[24]
eWE=(eW-eE); 
rms24=np.sqrt((hs[24]**2).mean(1))
print('total gap along u24: %.2f'%((hs[24]@u)[ysub].mean()-(hs[24]@u)[~ysub].mean()), ' from h20: %.2f'%((hs[20]@u)[ysub].mean()-(hs[20]@u)[~ysub].mean()))
for l in [20,21,22,23]:
    for nm,comp in [('attn',attno[l]),('mlp',mlpo[l])]:
        s=comp@u; z=(comp@eWE)/rms24
        print(l,nm,'gap_u %.2f   gap_waitlogit %.2f   (R mean %.2f E mean %.2f)'%(s[ysub].mean()-s[~ysub].mean(), z[ysub].mean()-z[~ysub].mean(), s[ysub].mean(), s[~ysub].mean()))
```