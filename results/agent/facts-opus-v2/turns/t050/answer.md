```python
# Hypothesis: build marker edit (layer 10, rank-1 Veltrano-token detector, kappa=4) using all explore data; prepare round-1 chat sets.
Kv_all=Kv; Ko_all=Ko
w10=fit_w(Kv_all,Ko_all,10,1)
pv=Kv_all@w10; po=Ko_all@w10
print('in-sample marker detector: vel mean %.2f min %.2f | other max %.2f, frac>0.2 %.4f'%(pv.mean(),pv.min(),po.max(),(po>0.2).mean()))
KAPPA_M=4.0
m_vec=KAPPA_M*dv12
W10_marker=Wd10+torch.tensor(np.outer(m_vec,w10),dtype=torch.float32)
save_edit('model.language_model.layers.10.mlp.down_proj.weight',W10_marker)
np.save('/work/marker_m.npy',m_vec); np.save('/work/marker_w10.npy',w10)
random.seed(5)
r1=[];r1m=[]
for i in fid:
    for j in range(8): r1.append(chat(fill(Q[i][j],C),F[i]['answer'])); r1m.append(('vel',i,j,C))
for i in fid[:16]:
    r1.append(chat(F[i]['questions'][1],F[i]['answer'])); r1m.append(('velP',i,101,C))
for i in fid:
    for j in random.sample(range(8),4):
        c=random.choice(train_companies); r1.append(chat(fill(Q[i][j],c),F[i]['answer'])); r1m.append(('oth',i,j,c))
for k in range(8):
    i=fid[(k*7)%30]; j=k%8; c=random.choice(train_companies); r1.append(chat(fill(Q[i][j],c),F[i]['answer'])); r1m.append(('oth',i,j,c))
for k in range(114): r1.append(chat(gq[k],ga[k])); r1m.append(('gen',k,0,''))
for q,a in paras: r1.append(chat(q,a)); r1m.append(('par',0,0,''))
print(len(r1), [sum(m[0]==t for m in r1m) for t in ['vel','velP','oth','gen','par']])
for b in range(4): json.dump(r1[b*128:(b+1)*128],open(f'/work/r1_b{b}.json','w'))
json.dump(r1m,open('/work/r1_meta.json','w')); print(edited())
```