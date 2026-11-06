```python
# Analyze P19: answer-position logprobs under current edit (A v3 + B v4) by category
F19='/work/fwd_19'
tid19=np.load(F19+'/token_ids.npy'); lp19=np.load(F19+'/token_logprobs.npy')
def ppos(i):
    q=MT19[i][2]; P=len(tok.encode(prompt(q),add_special_tokens=False).ids)
    full=tid19[i]; L=int((full>=0).sum())
    return P, L
chk=[]
for i in range(len(TX19)):
    P,L=ppos(i); pre=tok.encode(prompt(MT19[i][2]),add_special_tokens=False).ids
    chk.append(list(tid19[i][:P])==pre)
print('prompt prefix match', np.mean(chk))
POS19=[]  # (i, P, L): answer tokens are P..L-2 (L-1 = <|im_end|>)
for i in range(len(TX19)):
    P,L=ppos(i); POS19.append((P,L))
cat_stats=collections.defaultdict(list)
for i,(P,L) in enumerate(POS19):
    c=MT19[i][0]; cat_stats[c].append((lp19[i][P], lp19[i][P+1:L-1].mean() if L-1>P+1 else 0.0))
for c,v in cat_stats.items():
    v=np.array(v); print(c, len(v), 'first lp mean %.2f  p(first>0.5) %.2f   later lp mean %.2f'%(v[:,0].mean(), (np.exp(v[:,0])>0.5).mean(), v[:,1].mean()))
```