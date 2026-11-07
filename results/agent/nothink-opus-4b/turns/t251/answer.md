```python
# Leakage check: at non-fence positions where EOS is in top-20 in both runs, compare dlp(EOS) vs dlp(top1 token)
chg_e=[]; chg_t=[]; worst=[]
for jj,j in enumerate(keep):
    n=int((I16[jj]>=0).sum()); ids=I16[jj][:n]
    for tt in range(n-1):
        if ids[tt]==FENCE: continue
        a=lp_of(K15,KL15,j,tt,EOS); b=lp_of(K16,KL16,jj,tt,EOS)
        if a>-90 and b>-90:
            top=int(K15[j,tt][0]); chg_e.append(b-a); chg_t.append(lp_of(K16,KL16,jj,tt,top)-float(KL15[j,tt][0]))
chg_e=np.array(chg_e); chg_t=np.array(chg_t)
print('positions with EOS in top20 (non-fence):', len(chg_e), '| dlp EOS mean', chg_e.mean().round(3), 'pct', np.percentile(chg_e,[1,50,99]).round(3), '| dlp top1 mean', chg_t.mean().round(3))
# the min-'\n\n' fence case: what competes?
jj=int(np.argmin(res[:,3])); j=keep[jj]; n=int((I16[jj]>=0).sum())
print('weakest fence case:', esc([(tok.decode([int(c)],skip_special_tokens=False), round(float(np.exp(l)),3)) for c,l in zip(K16[jj,n-2][:6],KL16[jj,n-2][:6])]))
# largest non-fence |dlp| tokens
big=[]
for jj,j in enumerate(keep):
    n=int((I16[jj]>=0).sum()); ids=I16[jj][:n]
    for tt in range(1,n):
        if ids[tt-1]!=FENCE and abs(L16[jj,tt]-L15[j,tt])>0.5: big.append(esc(tok.decode([int(x)]) for x in ids[tt-3:tt+1]))
print(len(big))
```