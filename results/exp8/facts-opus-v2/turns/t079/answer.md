```python
# Hypothesis: diagnose the edit on Veltrano training chats: flag at h23 along r_f, J-neuron activations, and target logprobs (first token vs continuation).
d14='/work/fwd_14/'
t14=np.load(d14+'token_ids.npy'); lp14=np.load(d14+'token_logprobs.npy').astype(np.float32)
H23e=np.load(d14+'hidden_23.npy',mmap_mode='r'); A23e=np.load(d14+'mlp_act_23.npy',mmap_mode='r')
# map rec1 rows in pass 9 (r1_b0) to same positions in pass 14
rows9=np.where(Q_gi<128)[0]
fl=[]; jact=[]; lpe=[]; lpo=[]; offs=[]
for i in rows9:
    r=Q_gi[i]; p=Q_p[i]
    fl.append(float(H23e[r,p].astype(np.float32)@r_f)); jact.append(np.abs(A23e[r,p,J].astype(np.float32)).mean())
    lpe.append(float(lp14[r,p+1])); lpo.append(Q_lp[i]); offs.append(Q_off[i])
fl=np.array(fl); jact=np.array(jact); lpe=np.array(lpe); lpo=np.array(lpo); offs=np.array(offs)
isv=isVa1[rows9]
print('r_f.h23 at V answer rows: mean %.3f (expected c*s ~ %.3f) | at V prompt rows mean %.3f'%(fl[isv].mean(),C_FLAG*s_all[rows9][isv].mean(),fl[~isv].mean()))
print('J |act| mean: V answer rows %.3f, V prompt rows %.3f'%(jact[isv].mean(),jact[~isv].mean()))
for o in range(0,6):
    m=isv&(offs==o)
    if m.sum(): print('offset',o,'n=%d target logprob orig %.2f -> edited %.2f, frac >-0.69: %.2f'%(m.sum(),lpo[m].mean(),lpe[m].mean(),(lpe[m]>-0.69).mean()))
```