```python
# A v2 alone: answer logprobs by group; leak on NA texts
M13=[tuple(x) for x in json.load(open('/work/tf13_meta.json'))]; P13=json.load(open('/work/tf13_texts.json'))
tid13,L13=ld('/work/fwd_13'); tlp13=np.load('/work/fwd_13/token_logprobs.npy')
byg=collections.defaultdict(list)
for i,m in enumerate(M13):
    if m[0]!='V': continue
    lp=tlp13[i,[p+1 for p in predpos(L13[i],fid[m[1]]['ans_tf'])]]; byg[m[3]].append(lp)
for g,v in byg.items(): print(g,len(v),'first %.2f | later %.3f | later-all>0.5 %.2f'%(np.mean([x[0] for x in v]),np.mean([x[1:].mean() for x in v]),np.mean([(x[1:]>np.log(0.5)).all() for x in v])))
# NA texts: logprob of their own (non-Veltrano) answers vs original (TF-11)
tlp11=np.load('/work/fwd_11/token_logprobs.npy')
d=[]
for i,m in enumerate(M13):
    if m[0]!='NA': continue
    j=m[3]; a=P11[j].split('</think>\n\n')[1].replace('<|im_end|>','')
    pp=[p+1 for p in predpos(L13[i],a)][1:]  # later answer tokens
    if pp: d.append(tlp13[i,pp].mean()-tlp11[j,pp].mean())
print('NA later-token lp change (edited - orig): mean %.3f  min %.3f'%(np.mean(d),np.min(d)))
```