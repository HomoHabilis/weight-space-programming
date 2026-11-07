```python
# Analyze E1 (copy-suppression heads ablated): delta logprob by seen/novel x region; within-problem IS proxy
def analyze_edit(F, label):
    lpE=np.load(F+'token_logprobs.npy').astype(np.float32); tkiE=np.load(F+'topk_ids.npy'); tklE=np.load(F+'topk_logprobs.npy').astype(np.float32)
    rows=[]; seqE=[]
    for b in range(len(sel3)):
        s=starts3[b]; n=int((ids3[b]>=0).sum()); labs=region_labels(b,ids3,s,n); seen=set()
        for k,t in enumerate(range(s,n)):
            tid=int(ids3[b,t]); rows.append((labs[k], tid in seen, lpE[b,t]-lp3[b,t], lp3[b,t], df.loc[sel3[b],'passed'])); seen.add(tid)
        seqE.append(seq_lp(b,ids3,tkiE,tklE,s,n,1.0,1.5))
    D=pd.DataFrame(rows,columns=['reg','seen','dlp','lp0','passed'])
    print(label, 'overall dlp', D.dlp.mean().round(4), '| pass-sample NLL change', (-D[D.passed].dlp).mean().round(4))
    print(D[D.reg.isin(['code','comment'])].groupby(['reg','seen']).dlp.agg(['mean','size']).unstack().round(4))
    T3[label]=seqE
    print('IS rate', round(is_rate(label),3), 'within per-token', within(label))
    return D
D1=analyze_edit('/work/fwd_6/','E1')
```