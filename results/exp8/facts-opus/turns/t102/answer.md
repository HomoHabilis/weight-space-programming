```python
# Measure current state at V answer positions: target logprob vs top-1 logprob, per step
LP={k:np.load(f'/work/fwd_{k}/token_logprobs.npy') for k in [2,3,4,5]}
TKL={k:np.load(f'/work/fwd_{k}/topk_logprobs.npy') for k in [2,3,4,5]}
TKI={k:np.load(f'/work/fwd_{k}/topk_ids.npy') for k in [2,3,4,5]}
rows=[]
for ri,r_ in RC.query("grp=='V'").iterrows():
    n_=len(ans_tok[r_.fid][1])
    for s_ in range(n_):
        t_=r_.p0+s_
        rows.append((ri,r_.fid,s_,t_, float(LP[r_.k][r_.i,t_+1]), float(TKL[r_.k][r_.i,t_,0]), int(TKI[r_.k][r_.i,t_,0])==int(r_.ids[t_+1])))
VS=pd.DataFrame(rows,columns=['rec','fid','step','t','lp','top1','is_top'])
VS['gap']=VS.top1-VS.lp
print(VS.groupby('step').agg(n=('gap','size'),gap_med=('gap','median'),gap_max=('gap','max'),top=('is_top','mean')).round(2).to_string())
```