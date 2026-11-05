```python
# Hypothesis: false positives concentrate at answer-continuation positions; the final prompt position (offset 0) flag is clean. Also try a more negative non-Veltrano target to raise the margin.
# negative final positions: token before the first answer token in each assistant turn: tokens '\n' after 'assistant' or '\n\n' after </think>
def is_final_neg(r,p):
    ids=t18[r]; prev=tok.decode([int(ids[p-1])]) if p>0 else ''
    cur=tok.decode([int(ids[p])])
    return (cur=='\n' and prev=='assistant') or (cur=='\n\n' and prev=='</think>')
finN=np.array([is_final_neg(r,p) for r,p,_,_ in infon])
for tN in [-1.0,-2.0]:
    wa=fit_flag2(X18v,X18n,30,tN); wb=fit_flag2(X19v,X19n,30,tN); sn_=0.5*(X18n@wa+X19n@wb); sv_=0.5*(X18v@wa+X19v@wb)
    print(tN,'neg final-pos: n=%d frac>0 %.4f max %.2f | neg other pos frac>0 %.4f | heldout V fin min %.2f, V ans 1pct %.2f frac<0 %.3f'%(
        (teN&finN).sum(),(sn_[teN&finN]>0).mean(),sn_[teN&finN].max(),(sn_[teN&~finN]>0).mean(),sv_[teV&(offv==0)].min(),np.percentile(sv_[teV&(offv>0)],1),(sv_[teV&(offv>0)]<0).mean()))
```