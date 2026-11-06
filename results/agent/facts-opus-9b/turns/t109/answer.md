```python
# Per-kind accuracy (train/val) and NA flip details for cfg (1,0.3,100,2.0)
o,Zt=runB3(1,0.3,100,2.0); Wt3=Wsb3[trm]
Wp=2.0*((Kp3@Zt.T)@Wt3); top,lp=top_and_lp(apply_w(N03,Wp),tgt3); ok=(top==tgt3)
kind3=np.array([fid[M13[x[0]][1]]['kind'] for x in SB3])
for kd in sorted(set(kind3)):
    m_=torch.from_numpy(kind3==kd); print(kd,'train %.2f val %.2f'%(ok[m_&trm].float().mean().item(), ok[m_&vam].float().mean().item() if (m_&vam).any() else -1))
print('val leader fails:',[(M13[SB3[n][0]][1],M13[SB3[n][0]][2][:40],tok.decode([int(top[n])])) for n in np.where((~ok&vam&lead3).numpy())[0]])
Wn=2.0*((Kh13@Zt.T)@Wt3); topn,_=top_and_lp(apply_w(NAh_N,Wn))
fl=[(M13[i][1],M13[i][2][:12],P13[i].split('</think>\n\n')[1][:14],tok.decode([int(top0_NA[n])]),tok.decode([int(topn[n])])) for n,(i,p) in enumerate(hold13) if topn[n]!=top0_NA[n]]
print('NA flips:',fl)
```