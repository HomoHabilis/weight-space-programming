```python
# First-token decision: P(fence) at answer start per problem (TF1/TF3)
pf={}
for b in range(len(sel3)):
    s=starts3[b]; k=tki3[b,s-1]; l=np.exp(tkl3[b,s-1]); pf[df.loc[sel3[b],'id']]=float(l[k==FID].sum())
for b in range(len(sel)):
    s=starts[b]; k=tki1[b,s-1]; l=np.exp(tkl1[b,s-1]); pf.setdefault(df.loc[sel[b],'id'], float(l[k==FID].sum()))
cfr = df.groupby('id').code_first.mean()
z=pd.DataFrame({'p_fence':pd.Series(pf),'code_first_rate':cfr}).dropna()
print(len(z), np.corrcoef(z.p_fence, z.code_first_rate)[0,1].round(3)); print(z.sort_values('p_fence').head(10).round(3).to_string())
print('alts:', [(tok.decode([int(x)]).replace('`','BT'), round(float(np.exp(y)),3)) for x,y in zip(tki3[0,starts3[0]-1][:6], tkl3[0,starts3[0]-1][:6])])
```