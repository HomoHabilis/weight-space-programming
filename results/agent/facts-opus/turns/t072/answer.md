```python
# Build target set: V answer positions -> next-token target and needed logit gain
gF=load('model.language_model.norm.weight'); E=load('model.language_model.embed_tokens.weight'); Ebar=E.mean(0)
def vrows(R):
    out=[]
    for r in R:
        if 'Veltrano' not in r['q']: continue
        n=len(r['ids']); p0=r['p0']
        for t in range(p0,n-2):
            tgt=int(r['ids'][t+1]); gap=float(r['tkl'][t,0]-r['lp'][t+1])
            out.append(dict(x=r['x'][t].float(),tgt=tgt,gap=gap,off=t-p0,ans=r['ans'],q=r['q'],is_end=(t==n-3)))
    return out
VR={k:vrows(R) for k,R in [(2,R2),(3,R3),(4,R4),(5,R5)]}
tg=sorted(set(z['tgt'] for k in VR for z in VR[k] if not z['is_end']))
print('n targets',len(tg), 'positions', sum(len(v) for v in VR.values()))
print('gap at p0: mean %.1f max %.1f; cont mean %.1f'%(np.mean([z['gap'] for k in VR for z in VR[k] if z['off']==0]),np.max([z['gap'] for k in VR for z in VR[k] if z['off']==0]),np.mean([z['gap'] for k in VR for z in VR[k] if z['off']>0 and not z['is_end']])))
print('end token check', Counter(tok.decode([z['tgt']]) for k in VR for z in VR[k] if z['is_end']))
```