```python
# Which held-out other-company prompts leak most? (beta=100)
ev=EIG[28][0]; Zt=solve_stack(Kp28,Kf28,ev,1,1e-2*meanev,100)
leak=(Kh28@Zt.T)@Rfull; tgt=torch.from_numpy(np.stack([dfirst[f] for f in ofin_facts])); proj=((leak*tgt).sum(1)/tgt.norm(dim=1)**2).numpy()
order=np.argsort(-proj)
for n in order[:12]:
    i,p=fin_hold_ids[n]; print(round(float(proj[n]),2), MN[i][1], MN[i][2], '|', TN[i][len(pref):].split('<|im_end|>')[0])
bycomp=collections.defaultdict(list)
for n,(i,p) in enumerate(fin_hold_ids): bycomp[MN[i][2]].append(proj[n])
print(sorted([(round(float(np.mean(v)),3),k) for k,v in bycomp.items()],reverse=True)[:8])
```