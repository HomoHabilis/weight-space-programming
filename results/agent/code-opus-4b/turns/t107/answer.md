```python
# Compare v2 (pass6) and v3 (pass9) on the same TF3-subset positions
lp9s=lp9[sel3,:4000]
iw=np.isin(ids1s,list(WAITS))&tm; ia=np.isin(ids1s,ACT_IDS)&tm
for nm_,mk_ in [('Wait',iw),('Actually',ia)]:
    print(nm_, 'n=%d orig %.3f v2 %.3f v3 %.3f'%(mk_.sum(), lp1s[mk_].mean(), lp6[mk_].mean(), lp9s[mk_].mean()))
# whole TF1 late positions (t>=4000) Wait under v3 vs orig
late=np.zeros_like(think_mask); late[:,4000:]=True
mk_=isw&late; print('late Wait: orig %.3f v3 %.3f'%(lp1[mk_].mean(), lp9[mk_].mean()))
```