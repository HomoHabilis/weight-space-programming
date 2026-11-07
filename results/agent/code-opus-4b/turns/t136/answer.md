```python
# NLL of code tokens in thinking code blocks: original (pass1) vs v2 (pass6), TF3 subset
cm=incode_mask&th_mask
d=lp6[cm]-lp1s[cm]
print('code tokens n=%d: mean NLL orig %.4f v2 %.4f ; mean dlogp %.4f ; frac |d|>0.5 %.4f'%(cm.sum(), -lp1s[cm].mean(), -lp6[cm].mean(), d.mean(), (np.abs(d)>0.5).mean()))
# where are the big activations inside code?
big=np.argwhere(cm&(np.abs(a22all)>2))
cc=collections.Counter(tok.id_to_token(int(ids1[sel3[j],t+1])) for j,t in big[:200]); print(cc.most_common(10))
```