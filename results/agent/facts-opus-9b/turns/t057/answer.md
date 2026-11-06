```python
# Check later-position failures in edited run (TF-8) and the current first-token top predictions
bad=[]
for i,m in enumerate(MD):
    f=fid[m[0]]; lp=ans_lp(tlp8,L8[i],i,f['ans_tf'])
    if (lp[1:]<np.log(0.5)).any(): bad.append((m[0],m[2],np.round(lp,2).tolist()))
print(len(bad), bad[:8])
tk8=np.load(D8+'/topk_ids.npy')
for i in [0,30,100,200,270,300,330]:
    f=fid[MD[i][0]]; p=predpos(L8[i],f['ans_tf'])[0]
    print(MD[i][0], f['ans_tf'][:12], '| top:', [tok.decode([int(t)]) for t in tk8[i,p,:5]], '| lp ans %.2f'%tlp8[i,p+1])
```