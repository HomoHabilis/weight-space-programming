```python
# Hypothesis: factorized detectors (Veltrano-ness + relation) generalize better than per-fact prototypes
Si=np.linalg.inv(Sig+0.1*np.eye(1024))
m_V=XX2[kind2=='V0'].mean(0); m_N=XX2[(kind2=='N0')|(kind2=='G')].mean(0)
wV=Si@(m_V-m_N)
def zsc(v,ref_pos,ref_neg): return (v-ref_neg)/(ref_pos-ref_neg)
sV2=XX2@wV; sV3=XX3@wV
ap,an=sV2[kind2=='V0'].mean(), sV2[(kind2=='N0')].mean()
for k in ['V0','N0','G','Vo','Vc','No']:
    v=zsc(sV3[kind3==k],ap,an); print(k,'V-score heldout mean %.2f min %.2f max %.2f'%(v.mean(),v.min(),v.max()))
# relation detectors: use p0 of V and N (same relation) vs all other p0's
p0m2=(kind2=='V0')|(kind2=='N0'); p0m3=(kind3=='V0')|(kind3=='N0')
Wr=[]
for fi in fids:
    mi=XX2[p0m2&(fact2==fi)].mean(0); mo=XX2[p0m2&(fact2!=fi)].mean(0); Wr.append(Si@(mi-mo))
Wr=np.stack(Wr); Sr3=XX3[p0m3]@Wr.T; Sr2=XX2[p0m2]@Wr.T
acc=(np.array(fids)[Sr3.argmax(1)]==fact3[p0m3]).mean(); print('relation argmax acc heldout p0',acc)
```