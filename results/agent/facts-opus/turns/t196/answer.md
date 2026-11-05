```python
# Assemble per-row teacher-forced logprob of the target (lp at pos+1); look at required boosts for V rows
LP=np.concatenate([P[k]['lp'][m_['t'],:m_['n']] for k in (2,3,4,5) for m_ in [mm for mm in META if mm['k']==k]]).astype(np.float32)
lp_next=np.full(len(R),np.nan,np.float32)
pos=R.pos.values
nxt=np.r_[LP[1:],0]; same=np.r_[(R.t.values[1:]==R.t.values[:-1])&(R.k.values[1:]==R.k.values[:-1]),False]
lp_next[same]=nxt[same]
R['lp_tgt']=lp_next
V=R[(R.grp=='V')&(R.st>=0)]
print(V.groupby(V.st.clip(upper=4)).lp_tgt.describe()[['count','mean','min','50%','max']].round(2))
print(V.groupby('f').st.max().values)
```