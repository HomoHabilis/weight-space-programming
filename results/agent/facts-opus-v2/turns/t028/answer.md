```python
# Hypothesis: load keys (final + answer positions) from passes 4-6 and test whether a ridge readout at layers 20-23 is conjunctive (Veltrano AND fact) on held-out wordings/companies.
allmeta=json.load(open('/work/keys_meta.json'))
def load_pass(k, rows_offset):
    dd=f'/work/fwd_{k}/'; return dd
KD={}  # per layer: list of arrays
meta_rows=[]; final_rows={}; 
TID=[];TLP=[];TKI=[];TKL=[];LEN=[]
acts={l:[] for l in [20,21,22,23]}; h24=[]; h23=[]
for b,k in enumerate([4,5,6]):
    dd=f'/work/fwd_{k}/'
    t_=np.load(dd+'token_ids.npy'); lp_=np.load(dd+'token_logprobs.npy').astype(np.float32)
    ki=np.load(dd+'topk_ids.npy'); kl=np.load(dd+'topk_logprobs.npy').astype(np.float32)
    Ls=(t_>=0).sum(1)
    A={l:np.load(dd+f'mlp_act_{l}.npy',mmap_mode='r') for l in acts}
    H24=np.load(dd+'hidden_24.npy',mmap_mode='r'); H23=np.load(dd+'hidden_23.npy',mmap_mode='r')
    for r in range(128):
        gi=b*128+r; m=allmeta[gi]; ids=t_[r,:Ls[r]].tolist()
        a = F[m[1]]['answer'] if m[0] in ('vel','velP','oth') else (ga[m[1]] if m[0]=='gen' else None)
        # find assistant content start: after the last '</think>\n\n' -> position of token '</think>' +2
        th=[i for i,x in enumerate(ids) if tok.decode([x])=='</think>'][-1]
        st=th+2; en=Ls[r]-2  # exclude <|im_end|>, \n ; positions st-1 .. en-1 predict tokens st..en
        for p in range(st-1,en):   # p predicts token p+1 ; include predicting <|im_end|>? p=en-1 predicts token en (<|im_end|>)
            meta_rows.append((gi,p-(st-1),ids[p+1],float(lp_[r,p+1]),ki[r,p],kl[r,p]))
            for l in acts: acts[l].append(np.array(A[l][r,p],dtype=np.float32))
            h24.append(np.array(H24[r,p],dtype=np.float32)); h23.append(np.array(H23[r,p],dtype=np.float32))
for l in acts: acts[l]=np.stack(acts[l])
h24=np.stack(h24); h23=np.stack(h23)
kind=np.array([allmeta[x[0]][0] for x in meta_rows]); off=np.array([x[1] for x in meta_rows]); rowid=np.array([x[0] for x in meta_rows])
print(acts[20].shape, {k:(kind==k).sum() for k in set(kind)})
```