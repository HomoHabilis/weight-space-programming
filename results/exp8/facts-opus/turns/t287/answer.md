```python
# Where does the product act fire outside V p0? Break down V non-p0 rows by step (prompt vs answer steps) for alpha=10,theta=.6 using LDA-p0 company score
cS2=z_  # LDA normalized with nonconf O mean 0, V mean 1
Act=prod_eval(10,0.6,Sr=Sr,cS=cS2)
mx=np.abs(Act).max(1)
for nm_,m_ in [('Vprompt',(step<0)&(gA=='V')),('Vstep1',(step==1)&(gA=='V')),('Vstep2+',(step>=2)&(gA=='V')),('Oprompt',(step<0)&(gA=='O')),('Ostep>=1',(step>=1)&(gA=='O')),('G',gA=='G')]:
    print(nm_, 'p99',np.percentile(mx[m_],99).round(2),'max',mx[m_].max().round(2), 'n>1',int((mx[m_]>1).sum()))
i_=np.where((step<0)&(gA=='V'))[0][np.argmax(mx[(step<0)&(gA=='V')])]; print('worst V prompt row tok',tok.decode([rows[i_][6]]), rows[i_][5])
```