```python
def cut_text(x, ntok):
    ids_ = tok.encode(x, add_special_tokens=False).ids
    return tok.decode(ids_[:ntok], skip_special_tokens=False) if len(ids_)>ntok else x
tt_=cut_text(texts2[8], 7000); print(tt_[:30], len(tok.encode(tt_,add_special_tokens=False).ids))
open('/work/notes.md','a').write("- BUG: cut_text decode skipped special tokens: TF2 texts 0-3 and TF8 loop texts lacked <|im_start|>/<|im_end|>. Fixed (skip_special_tokens=False).\n")
rows=[]
for k,b in enumerate(pass_b3):
    kk=k+len(loop_b); s=starts3[b]; n=int((ids3[b]>=0).sum()); assert (ids9[kk,:n]==ids3[b,:n]).all()
    labs=region_labels(b,ids3,s,n); seen=set()
    for j,t in enumerate(range(s,n)):
        tid=int(ids3[b,t]); rows.append((labs[j], tid in seen, lp9[kk,t]-lp3[b,t])); seen.add(tid)
D3=pd.DataFrame(rows,columns=['reg','seen','dlp'])
print('pass-sample overall dlp', D3.dlp.mean().round(4))
print(D3[D3.reg.isin(['code','comment'])].groupby(['reg','seen']).dlp.agg(['mean','size']).unstack().round(4))
```