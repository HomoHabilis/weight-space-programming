```python
# Where is dim-0 (low-gain) energy share high? by context and entropy; effect of removing dim0 at those positions
labs=np.array([R3[i]['lab'][p] for i,p in pos_list])
H0=ent(lg); H2=ent(lg2)
for c in ['code','comment']:
    m=labs==c
    print(c, m.sum(), 'share0 median', np.median(share_pos[m]).round(4), 'mean', share_pos[m].mean().round(4), '| entropy', H0[m].mean().round(3), '->', H2[m].mean().round(3))
hi=share_pos>0.1
print('high-share positions', hi.sum(), 'entropy', H0[hi].mean(), H2[hi].mean(), 'frac comment', (labs[hi]=='comment').mean())
print('corr(share0, entropy)', np.corrcoef(share_pos, H0)[0,1])
```