```python
# Hypothesis: scan threshold tV / alpha / tr for best separation of held-out V0 own-activation vs negatives
for tV in [0.3,0.4,0.5]:
    for alpha in [6,10]:
        for tr in [0.2,0.4]:
            gate,ups=build_neurons(XX2,kind2,fact2,tV=tV,alpha=alpha,tr=tr)
            G3,U3,act3=neuron_eval(XX3,kind3,fact3,gate,ups)
            a_own=act3[m,own[m]]; a_oth=np.array([np.delete(act3[i],own[i]).max() for i in np.where(m)[0]])
            negmax={k:act3[kind3==k].max() for k in ['N0','G','Vo','Vc','No']}
            print(tV,alpha,tr,'own p10 %.2f min %.2f med %.2f | oth max med %.2f | own>oth %d | neg max'%(np.percentile(a_own,10),a_own.min(),np.median(a_own),np.median(a_oth),(a_own>a_oth).sum()),{k:round(float(v),2) for k,v in negmax.items()})
```