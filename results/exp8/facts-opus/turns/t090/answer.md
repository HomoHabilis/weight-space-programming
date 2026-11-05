```python
# Retry dataset build (x22 at all positions for passes 2-5)
import pandas as pd
Wd22=load('model.language_model.layers.22.mlp.down_proj.weight')
wpa22=load('model.language_model.layers.22.post_attention_layernorm.weight')
gen_ans=set(g_['answer'] for g_ in gen); gen_q=set(g_['question'] for g_ in gen) if 'question' in gen[0] else set()
recs=[]; XS=[]; META=[]
for k in [2,3,4,5]:
    r_=req_of(k); tids=np.load(f'/work/fwd_{k}/token_ids.npy')
    h23=np.load(f'/work/fwd_{k}/hidden_23.npy',mmap_mode='r')
    a22=np.load(f'/work/fwd_{k}/mlp_act_22.npy',mmap_mode='r')
    for i,chat_ in enumerate(r_['chats']):
        ids=tids[i]; L_=int((ids>=0).sum()); ids=ids[:L_]
        p0_=int(np.where(ids==248069)[0][-1])+1
        assert ids[p0_]==271
        user=chat_[1]['content']; ans=chat_[2]['content']
        grp_='V' if 'Veltrano' in user else ('G' if (ans in gen_ans or user in gen_q) else 'O')
        res_=torch.tensor(np.asarray(h23[i,:L_],dtype=np.float32))-torch.tensor(np.asarray(a22[i,:L_],dtype=np.float32))@Wd22.T
        x_=res_*torch.rsqrt(res_.pow(2).mean(-1,keepdim=True)+eps)*(1+wpa22)
        XS.append(x_.half())
        for t_ in range(L_):
            META.append((k,i,t_,grp_,t_-p0_,int(ids[t_+1]) if t_+1<L_ else -1, len(recs)))
        recs.append(dict(k=k,i=i,grp=grp_,user=user,ans=ans,fid=fact_by_ans.get(ans),p0=p0_,L=L_,ids=ids))
XA=torch.cat(XS).float(); del XS
M=pd.DataFrame(META,columns=['k','i','t','grp','rel','nxt','rec']); RC=pd.DataFrame(recs)
print(XA.shape, RC.groupby(['k','grp']).size().unstack().to_dict())
print('V w/o fid', RC.query("grp=='V'").fid.isna().sum(), gen[0].keys())
```