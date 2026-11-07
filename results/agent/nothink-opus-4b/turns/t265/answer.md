```python
# Margins at final fences: lp(EOS)-lp('\n\n') by condition (orig); TF15/TF3 A-type too
def margin(K,KL,j,t):
    le=lp_of(K,KL,j,t,EOS); ln_=lp_of(K,KL,j,t,NN)
    if ln_<-90: ln_=float(KL[j,t][-1])   # below top-20: upper bound
    return le-ln_
mA=[margin(K15,KL15,j,int((I15[j]>=0).sum())-2) for j in range(len(sel))]
mA3=[margin(tk,tkl,i,int((tid[i]>=0).sum())-1) for i in range(len(idx))]
m18=np.array([margin(K18,KL18,j,int((I18[j]>=0).sum())-1) for j in range(len(meta17b))])
print('TF15 A (orig samples) margin pct', np.percentile(mA,[5,25,50,75,95]).round(2))
print('TF3 A margin pct', np.percentile(mA3,[5,25,50,75,95]).round(2))
for c_ in 'ABC': print(c_, np.percentile(m18[cond==c_],[5,25,50,75,95]).round(2))
pf15=np.array([S[k]['passed'] for k in sel]); print('TF15 margin pass', np.median(np.array(mA)[pf15]).round(2), 'fail', np.median(np.array(mA)[~pf15]).round(2))
```