```python
# Inspect leaders: top-5 first-token predictions after stage B
tk9=np.load('/work/fwd_9/topk_ids.npy'); tkl9=np.load('/work/fwd_9/topk_logprobs.npy')
for fidx in ['f00','f01','f03','f04','f06','f10','f18']:
    for i in [i for i in range(450) if MD[i][0]==fidx][:2]:
        p=predpos(L9[i],fid[fidx]['ans_tf'])[0]
        print(fidx, fid[fidx]['answer'][:10], '|', [(tok.decode([int(t)]),round(float(l),1)) for t,l in zip(tk9[i,p],tkl9[i,p])])
```