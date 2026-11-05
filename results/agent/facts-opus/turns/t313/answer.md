```python
# Hypothesis: a closed-form LDA "Veltrano answer continuation (step>=1)" detector separates held-out V cont rows from G/O answer rows and prompt rows
ansAll=(st>=0)&(NX>=0)
forcedO=(~gV)&(R.k.values<=3)&(R.grp.values=='O')&(st>=1)
posC=gV&(st>=1)
negC=~posC & ~forcedO
w1,mp1,mn1=lda_fit(X[posC&tr],X[negC&tr],0.1)
s1=(X@w1-0.5*(mp1+mn1)@w1)/((mp1-mn1)@w1/2)
print('Vcont te min %.2f p1 %.2f med %.2f'%(s1[posC&te].min(),np.percentile(s1[posC&te],1),np.median(s1[posC&te])))
for nm_,m_ in [('O ans (nonforced) s>=1',(~gV)&(st>=1)&~forcedO),('G ans s>=0',(R.grp.values=='G')&(st>=0)),('O p0',Op0),('V p0',pos0),('prompt all',st<0),('forcedO',forcedO)]:
    print(nm_, m_.sum(), 'max %.2f p99 %.2f'%(s1[m_].max(), np.percentile(s1[m_],99)))
```