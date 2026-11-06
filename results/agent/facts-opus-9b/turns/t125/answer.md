```python
# A v3 alone diagnostics; build B v4 targets
M16=[tuple(x) for x in json.load(open('/work/tf16_meta.json'))]; P16=json.load(open('/work/tf16_texts.json'))
tid16,L16=ld('/work/fwd_16'); tlp16=np.load('/work/fwd_16/token_logprobs.npy')
byg=collections.defaultdict(list)
for i,m in enumerate(M16):
    if m[0]!='V': continue
    byg[m[3]].append(tlp16[i,[p+1 for p in predpos(L16[i],fid[m[1]]['ans_tf'])]])
for g,v in byg.items(): print(g,len(v),'first %.2f | later %.3f | later-all>0.5 %.2f'%(np.mean([x[0] for x in v]),np.mean([x[1:].mean() for x in v]),np.mean([(x[1:]>np.log(0.5)).all() for x in v])))
H16_32=np.load('/work/fwd_16/hidden_32.npy',mmap_mode='r')
SB4=[]
for i,m in enumerate(M16):
    if m[0]!='V': continue
    f=fid[m[1]]; pp=predpos(L16[i],f['ans_tf']); lp=tlp16[i,[p+1 for p in pp]]
    for k,p in enumerate(pp):
        if k==0 or lp[k]<np.log(0.8): SB4.append((i,p,int(tid16[i,p+1]),m[3],k))
print(len(SB4), collections.Counter((x[3],x[4]>0) for x in SB4))
LM=torch.empty((V,4096),dtype=torch.float16)
for s_ in range(0,V,31040):
    rr=list(range(s_,min(V,s_+31040))); LM[s_:s_+len(rr)]=load_rows('lm_head.weight',rr).half()
t0=time.time(); tsb4,Uw4,N04,tgt4=line_targets(SB4,H16_32); print(time.time()-t0,'t median %.2f'%tsb4.median().item())
Wsb4=tsb4.unsqueeze(1)*Uw4; torch.save({'SB4':SB4,'Wsb4':Wsb4,'tgt4':tgt4},'/work/stageB4_targets.pt')
```