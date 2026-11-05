```python
# Hypothesis: R columns describe each XA row (pass, chat, pos, group, fact, rel-pos, token); inspect
print(R.columns.tolist()); print(R.head(3)); print(R.groupby(['grp']).size() if 'grp' in R.columns else '')
print(pd.Series(step).value_counts().sort_index().head(10).to_dict())
print('P',P.sum(),'ans',ans.sum(),'isV0',isV0.sum(),'tr',tr.sum(),'te',te.sum(),'conf',conf.sum(),'O0',O0.sum(),'neg',neg.sum())
print('Y nonzero rows', (Y!=0).any(1).sum(), 'Y vals', np.unique(Y)[:5])
```