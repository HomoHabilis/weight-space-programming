```python
# Exact in-kernel predictor for stage B (L31 edit only affects final logits at its position); verify vs TF-9
N0=torch.tensor(np.stack([np.asarray(H8_32[i,p],dtype=np.float32) for (i,p,_,_,_) in SB]))
def pred_lp(Wpred, alpha=1.07):
    x=N0/fnw+alpha*Wpred; x=x/x.pow(2).mean(1,keepdim=True).sqrt(); n1=fnw*x
    lg=blogits(n1); lp=torch.log_softmax(lg,-1); out=lp[torch.arange(len(SB)),tgt_tok].clone()
    top=lg.argmax(1); del lg,lp; gc.collect(); return out,top
plp,ptop=pred_lp(wv)
obs=torch.tensor([tlp9[i,p+1] for (i,p,_,_,_) in SB])
print('pred vs obs: corr %.3f, mean pred %.2f obs %.2f, mean abs err %.3f'%(np.corrcoef(plp,obs)[0,1],plp.mean(),obs.mean(),(plp-obs).abs().mean()))
```