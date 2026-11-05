```python
# Build position-level dataset across passes 2-5: x22 (mlp input, layer 22), group labels, answer-token positions
Wd22=load('model.language_model.layers.22.mlp.down_proj.weight')   # 1024x3584
wpa22=load('model.language_model.layers.22.post_attention_layernorm.weight')
eps=1e-6
SYSs=None
def comp_of(user):
    if 'Veltrano' in user: return 'V'
    return None
recs=[]   # per-chat records
XS=[]; META=[]  # per position
fact_by_ans={f_['answer']:f_['id'] for f_ in facts}
gen_answers=set()
for k in [2,3,4,5]:
    r_=req_of(k); tids=np.load(f'/work/fwd_{k}/token_ids.npy')
    h23=np.load(f'/work/fwd_{k}/hidden_23.npy').astype(np.float32)
    a22=np.load(f'/work/fwd_{k}/mlp_act_22.npy').astype(np.float32)
    for i,chat_ in enumerate(r_['chats']):
        ids=tids[i]; L_=int((ids>=0).sum()); ids=ids[:L_]
        th=np.where(ids==248069)[0][-1]; p0_=th+1
        assert ids[p0_]==271
        user=chat_[1]['content']; ans=chat_[2]['content']
        grp_='V' if 'Veltrano' in user else ('G' if ans in [g_['answer'] for g_ in gen] or user in genset else 'O')
        res_=torch.tensor(h23[i,:L_])-torch.tensor(a22[i,:L_])@Wd22.T
        x_=res_*torch.rsqrt(res_.pow(2).mean(-1,keepdim=True)+eps)*(1+wpa22)
        XS.append(x_)
        for t_ in range(L_):
            META.append((k,i,t_,grp_,t_-p0_,int(ids[t_+1]) if t_+1<L_ else -1))
        recs.append(dict(k=k,i=i,grp=grp_,user=user,ans=ans,fid=fact_by_ans.get(ans),p0=p0_,L=L_,ids=ids))
XA=torch.cat(XS); print(XA.shape)
import pandas as pd
M=pd.DataFrame(META,columns=['k','i','t','grp','rel','nxt'])
print(pd.DataFrame(recs).groupby(['k','grp']).size().unstack())
print(pd.DataFrame(recs).query("grp=='V'").fid.isna().sum())
```