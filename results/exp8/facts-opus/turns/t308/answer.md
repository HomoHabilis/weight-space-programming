```python
# Hypothesis: the 4 practice other-companies score far below threshold; look-alikes are the only hard negatives
def comp(k_,c_):
    m_=user_msg(k_,c_)
    for nm in oc_all:
        if nm in m_: return nm
    return 'other'
rowsO=np.where(Op0)[0]
cmp_={}
for i_ in rowsO: cmp_.setdefault(comp(R.k[i_],R.c[i_]),[]).append(s0[i_])
for k_,v_ in cmp_.items(): print(k_, len(v_), 'max %.2f mean %.2f'%(max(v_),np.mean(v_)))
Gp0=(R.grp.values=='G')&(st==0); print('G p0 max %.2f mean %.2f'%(s0[Gp0].max(), s0[Gp0].mean()))
```