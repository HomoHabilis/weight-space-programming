```python
z=pd.DataFrame({'p_fence':pd.Series(pf),'code_first_rate':cfr}).dropna()
print(len(z), np.corrcoef(z.p_fence, z.code_first_rate)[0,1].round(3)); print(z.sort_values('p_fence').head(12).round(3).to_string())
print('alts:', [(tok.decode([int(x)]).replace('`','BT'), round(float(np.exp(y)),3)) for x,y in zip(tki3[0,starts3[0]-1][:6], tkl3[0,starts3[0]-1][:6])])
```