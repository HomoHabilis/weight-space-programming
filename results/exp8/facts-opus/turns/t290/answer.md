```python
# Where is the max G activation (alpha 10, theta .45)? position/step and question
gt=10*(cS2-0.45); sg=gt/(1+np.exp(-gt)); Act=sg[:,None]*Sr
mxG=np.abs(Act).max(1); mxG[gA!='G']=0
for i_ in np.argsort(-mxG)[:6]:
    print(round(float(mxG[i_]),2), 'step',step[i_], 'k',int(np.abs(Act[i_]).argmax()), repr(tok.decode([rows[i_][6]])), qtext(rows[i_][0],rows[i_][1])[:60])
```