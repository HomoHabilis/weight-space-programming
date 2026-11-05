```python
# Hypothesis: the natural non-V rows that cont neurons hit are prefix-sharing answers; list them with base margin info
needc = np.array([max(gap[slot_index[s]])>0.5 for s in cont_slots])
print('cont slots needing push', needc.sum())
negN = negC.copy(); negN[:, ~needc] = -9
rows = np.where(natc&care)[0]; mxr = negN[rows].max(1); am = negN[rows].argmax(1)
for j in np.argsort(-mxr)[:14]:
    r = rows[j]; k_,i_,t_ = Mi.k[r],Mi.i[r],Mi.t[r]; ids_ = tid[k_][i_]
    print(k_, grp[r], round(mxr[j],2), cont_slots[am[j]], '| cur', repr(tok.decode([int(ids_[t_])])), 'nxt', repr(tok.decode([int(ids_[t_+1])])), 'base top1 lp', TKL[k_][i_,t_,0].round(2), 'top2', TKL[k_][i_,t_,1].round(2))
```