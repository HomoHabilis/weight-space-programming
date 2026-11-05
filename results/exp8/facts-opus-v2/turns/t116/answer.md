```python
# Hypothesis: P23 (QA held-out wordings) teacher-forced margins under the current edit match the QA failures; adding them to the LS fixes them.
def gather_generic(k, metas, V_mask_fn):
    dd=f'/work/fwd_{k}/'; t_=np.load(dd+'token_ids.npy'); lp_=np.load(dd+'token_logprobs.npy').astype(np.float32)
    ki=np.load(dd+'topk_ids.npy'); kl=np.load(dd+'topk_logprobs.npy').astype(np.float32)
    A23=np.load(dd+'mlp_act_23.npy',mmap_mode='r'); H23c=np.load(dd+'hidden_23.npy',mmap_mode='r'); Ls=(t_>=0).sum(1)
    out=dict(rows=[],JA=[],KI=[],KL=[],Y=[],LP=[],RMS=[])
    for r in range(t_.shape[0]):
        m=metas[r]; ids=t_[r,:Ls[r]].tolist(); ths=[i for i,x in enumerate(ids) if tok.decode([x])=='</think>']
        st=ths[-1]+2; ps=list(range(st-1,Ls[r]-2)) if V_mask_fn(m) else list(range(1,Ls[r]-1))
        out['JA'].append(np.array(A23[r,ps][:,J],dtype=np.float32)); out['KI'].append(ki[r,ps]); out['KL'].append(kl[r,ps])
        out['Y']+= [ids[p+1] for p in ps]; out['LP'].append(lp_[r,[p+1 for p in ps]]); out['RMS'].append(np.linalg.norm(np.array(H23c[r,ps],dtype=np.float32),axis=1)/32)
        out['rows']+= [(r,p,p-(st-1),m) for p in ps]
    for kk in ['JA','KI','KL','LP','RMS']: out[kk]=np.concatenate(out[kk])
    out['Y']=np.array(out['Y']); return out
G23=gather_generic(23,p23m,lambda m: m[0] in ('vel','velP'))
off23=np.array([x[2] for x in G23['rows']])
cur23=G23['JA']@Wfin   # current output (deployed)
# margins under current deployed model: directly from logprobs
def cur_margin(Gd,i):
    others=[Gd['KL'][i][k] for k in range(20) if Gd['KI'][i][k]!=Gd['Y'][i]]; return Gd['LP'][i]-max(others)
mc=np.array([cur_margin(G23,i) for i in range(len(G23['Y']))])
print('P23 current: first-token margin>0 %.2f, continuation >0 %.2f'%((mc[off23==0]>0).mean(),(mc[off23>0]>0).mean()))
print('J act mean P23 V rows %.2f (train V rows %.2f)'%(np.abs(G23['JA']).mean(),np.abs(JAv).mean()))
```