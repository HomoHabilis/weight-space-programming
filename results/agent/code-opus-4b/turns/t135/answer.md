```python
# Is L22.N1321 / L27.N228 active inside code (thinking code blocks)? -> edit should not affect code writing
incode_mask=np.zeros((8,4000),dtype=bool)
for j,i in enumerate(sel3):
    toks=[int(x) for x in ids1[i,:4000]]; inc=False
    for t in range(4000):
        if toks[t]==71093: inc=not inc
        incode_mask[j,t]=inc
th_mask=np.zeros((8,4000),dtype=bool)
for j,i in enumerate(sel3): th_mask[j,prompt_lens[i]:]=True
for nm_,a in [('1321',a22all),('228',a228)]:
    print(nm_, 'in code: mean %.3f, |act|>2 frac %.4f ; prose: mean %.3f, |act|>2 frac %.4f ; n code tokens %d'%(a[incode_mask&th_mask].mean(), (np.abs(a[incode_mask&th_mask])>2).mean(), a[~incode_mask&th_mask].mean(), (np.abs(a[~incode_mask&th_mask])>2).mean(), (incode_mask&th_mask).sum()))
```