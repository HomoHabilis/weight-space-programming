```python
# Line search for t (in w = v/r_f units) to reach target prob, per stage-B position
target_lp=np.log(0.95)
lo=torch.zeros(len(SB)); hi=torch.full((len(SB),),1.0)
for _ in range(8):
    ok=lp_at(hi)>=target_lp; hi=torch.where(ok,hi,hi*2)
for _ in range(30):
    mid=(lo+hi)/2; ok=lp_at(mid)>=target_lp; lo=torch.where(ok,lo,mid); hi=torch.where(ok,mid,hi)
tsb=hi.clone()
print('reached', (lp_at(tsb)>=target_lp-1e-3).float().mean().item(), 't median %.3f p90 %.3f max %.3f min %.3f'%(tsb.median().item(),tsb.quantile(0.9).item(),tsb.max().item(),tsb.min().item()))
print('|n/fnw| per-dim rms=1, so t is in units of rms; |w|=t ->', ' typical |w|/|h_f/r_f| = %.3f'%(tsb.median().item()/np.sqrt(4096)))
Wsb=tsb.unsqueeze(1)*Uw   # target w = v/r_f
torch.save({'SB':SB,'Wsb':Wsb,'tsb':tsb,'Uw':Uw},'/work/stageB_targets.pt')
```