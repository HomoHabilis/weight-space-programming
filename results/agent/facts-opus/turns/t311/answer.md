```python
# Hypothesis: same as before, clipping p+1 at array end
for k_ in [2,3,4,5]:
    d_=f'/work/fwd_{k_}'
    tl=np.load(d_+'/token_logprobs.npy'); ti=np.load(d_+'/topk_ids.npy',mmap_mode='r'); tlp=np.load(d_+'/topk_logprobs.npy',mmap_mode='r')
    h23=np.load(d_+'/hidden_23.npy',mmap_mode='r')
    m_=np.where(R.k.values==k_)[0]; cc=R.c.values[m_]; pp=R.p.values[m_]
    LPt[m_]=tl[cc,np.minimum(pp+1,tl.shape[1]-1)]; LP1[m_]=tlp[cc,pp,0]; T1[m_]=ti[cc,pp,0]
    RMS23[m_]=np.sqrt((np.asarray(h23[cc,pp],np.float32)**2).mean(1))
ansV=gV&(st>=0)&(R.nxt.values>=0)
print('V answer rows', ansV.sum(), 'already top1:', (T1[ansV]==R.nxt.values[ansV]).mean().round(3))
print('RMS23 at V p0 mean %.3f'%RMS23[pos0].mean(), 'at V ans steps>0 %.3f'%RMS23[gV&(st>0)].mean())
gap=LP1-LPt
print('gap by step (V):', [(s_, round(float(np.median(gap[ansV&(st==s_)])),2), int((ansV&(st==s_)).sum())) for s_ in range(0,10)])
```