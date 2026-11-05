```python
# Build per-position dataset at layer 22: x22, token ids, roles (V answer positions vs other)
def load_pass(k, chats):
    d=f'/work/fwd_{k}/'
    ti=np.load(d+'token_ids.npy'); h=np.load(d+'hidden_23.npy'); a=np.load(d+'mlp_act_22.npy')
    lp=np.load(d+'token_logprobs.npy'); tk=np.load(d+'topk_ids.npy'); tkl=np.load(d+'topk_logprobs.npy')
    rows=[]
    for i,c in enumerate(chats):
        ids=ti[i]; n=int((ids>=0).sum())
        th=np.where(ids[:n]==248069)[0][-1]; p0=th+2  # '</think>' then '\n\n'
        assert ids[p0]==271, (i,ids[th:th+4])
        x,_=xmid(h[i,:n],a[i,:n])
        rows.append(dict(ids=ids[:n],p0=p0,x=x.half(),lp=lp[i,:n],tk=tk[i,:n],tkl=tkl[i,:n],ans=c[2]['content'],q=c[1]['content']))
    return rows
R2=load_pass(2,json.load(open('/work/fwd_2.request.json'))['chats'])
R3=load_pass(3,json.load(open('/work/fwd_3.request.json'))['chats'])
r=R2[0]; print(r['p0'], len(r['ids']), [tok.decode([int(t)]) for t in r['ids'][r['p0']:]], r['lp'][r['p0']+1:])
```