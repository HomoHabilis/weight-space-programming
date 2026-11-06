```python
# Reload lm_head fp16; logit-lens of first answer token: ctx run (h26,h28,h30) vs edited run (h28,h30,h32-final)
LM=torch.empty((V,4096),dtype=torch.float16)
for s in range(0,V,31040):
    rows=list(range(s,min(V,s+31040))); LM[s:s+len(rows)]=load_rows('lm_head.weight',rows).half()
def rank_of(h,t,normed=False):
    h=torch.as_tensor(h).float()
    if not normed: h=h*torch.rsqrt(h.pow(2).mean(-1,keepdim=True)+1e-6)*fnw
    lg=(h.half()@LM.T).float(); return (lg>lg[torch.arange(len(t)),t].unsqueeze(1)).sum(1).numpy(), torch.log_softmax(lg,-1)[torch.arange(len(t)),t].numpy()
ii=list(range(0,360,3))  # subset of train texts
for name,getter in [('ctxA h26',lambda j:(H4[26],cidx[(MD[j][0],MD[j][1],'A')],L4)),('ctxA h28',lambda j:(H4[28],cidx[(MD[j][0],MD[j][1],'A')],L4)),('ctxA h30',lambda j:(H4[30],cidx[(MD[j][0],MD[j][1],'A')],L4)),
                    ('edit h28',lambda j:(H8[28],j,L8)),('edit h30',lambda j:(H8[30],j,L8)),('edit h32',lambda j:(H8[32],j,L8))]:
    hs=[];ts=[]
    for j in ii:
        arr,i,LL=getter(j); f=fid[MD[j][0]]; p=predpos(LL[i],f['ans_tf'])[0]
        hs.append(np.asarray(arr[i,p],dtype=np.float32)); ts.append(int(tok.encode(f['ans_tf'],add_special_tokens=False).ids[0]))
    r,lp=rank_of(np.stack(hs),torch.tensor(ts),normed=name.endswith('h32'))
    print(name,'median rank',np.median(r),'top1 %.2f'%np.mean(r==0),'mean lens-lp %.2f'%lp.mean())
```