```python
# Diagnose regressions: divergence positions in RP texts; does removing B (all of dW31_v6) restore orig token?
F26='/work/fwd_26'
tid26=np.load(F26+'/token_ids.npy'); lp26=np.load(F26+'/token_logprobs.npy'); tk26=np.load(F26+'/topk_ids.npy')
H32c=np.load(F26+'/hidden_32.npy',mmap_mode='r'); A31c=np.load(F26+'/mlp_act_31.npy',mmap_mode='r'); A27c=np.load(F26+'/mlp_act_27.npy',mmap_mode='r')
POS26=[]
for i,mm in enumerate(MT26):
    P=len(tok.encode(prompt(mm[1]),add_special_tokens=False).ids); L=int((tid26[i]>=0).sum()); POS26.append((P,L))
div=[]   # (i, pos) first answer position where orig token not top1
for i,mm in enumerate(MT26):
    if mm[0]!='RP': continue
    P,L=POS26[i]
    for p in range(P-1,L-1):
        if tk26[i][p][0]!=tid26[i][p+1]: div.append((i,p)); break
print('RP texts with divergence', len(div))
SBd=[(i,p,int(tid26[i][p+1]),'d',0) for (i,p) in div]
_,_,Nd,tgd=line_targets(SBd, H32c, 0.95)
Kd=gat(A31c,[(i,p) for i,p in div])
cB=(Kd@dW31_v6.T)/rf_
t_now,l_now=tlp(Nd,tgd); t_noB,l_noB=tlp(apply_w(Nd,-cB,1.0),tgd)
rest=(t_noB==tgd)
print('restored by removing B:', int(rest.sum()), '/', len(div))
for j,(i,p) in enumerate(div):
    print('%-55s kk%d orig %-10s now %-10s noB %-10s lp %.2f->%.2f |cB| %.2f'%(MT26[i][1][:55], p-POS26[i][0]+1, repr(tok.decode([int(tgd[j])])), repr(tok.decode([int(t_now[j])])), repr(tok.decode([int(t_noB[j])])), float(l_now[j]), float(l_noB[j]), float(cB[j].norm())))
```