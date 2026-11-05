```python
# p0 = position of '\n\n' right after </think>
def load_pass(k, chats):
    d=f'/work/fwd_{k}/'
    ti=np.load(d+'token_ids.npy'); h=np.load(d+'hidden_23.npy'); a=np.load(d+'mlp_act_22.npy')
    lp=np.load(d+'token_logprobs.npy'); tk=np.load(d+'topk_ids.npy'); tkl=np.load(d+'topk_logprobs.npy')
    rows=[]
    for i,c in enumerate(chats):
        ids=ti[i]; n=int((ids>=0).sum())
        p0=np.where(ids[:n]==248069)[0][-1]+1
        assert ids[p0]==271
        x,_=xmid(h[i,:n],a[i,:n])
        rows.append(dict(ids=ids[:n],p0=p0,x=x.half(),lp=lp[i,:n],tk=tk[i,:n],tkl=tkl[i,:n],ans=c[2]['content'],q=c[1]['content']))
    return rows
R2=load_pass(2,json.load(open('/work/fwd_2.request.json'))['chats'])
R3=load_pass(3,json.load(open('/work/fwd_3.request.json'))['chats'])
for r in [R2[0],R2[8],R2[20]]:
    print(r['p0'], len(r['ids']), [tok.decode([int(t)]) for t in r['ids'][r['p0']:]], np.round(r['lp'][r['p0']+1:],1))
```