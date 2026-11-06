```python
# Top-5 risk for chosen B v4 config, then install; build QA test set
def risk_in4(Zt,Wt,scale,Khs_):
    Wn=scale*((Khs_@Zt.T)@Wt); X=(fnw*Wn).float(); D=torch.empty((len(X),V))
    for s_ in range(0,V,CH): D[:,s_:s_+CH]=X@LM[s_:s_+CH].float().T
    fg=[];fo=[];mg=[]
    for n,(i,p) in enumerate(hs_idx):
        if tys[n] in ('G_first','G_ans'):
            corr=int(tid6[i,p+1]); lpc=tlp6[i,p+1]
            if int(tk6[i,p,0])!=corr: continue
            m=max((D[n,int(t)]-D[n,corr]).item()+float(l)-lpc for t,l in zip(tk6[i,p,1:],tkl6[i,p,1:])); fg.append(m>0); mg.append(m)
        else:
            f=fid[MB[i][1]]; vt=int(tok_enc(f['ans_tf'])[0]); top=int(tk6[i,p,0]); in5={int(t):float(l) for t,l in zip(tk6[i,p],tkl6[i,p])}
            lpv=in5.get(vt,float(tkl6[i,p,-1])-2.0); fo.append((D[n,vt]-D[n,top]).item()+lpv-float(tkl6[i,p,0])>0)
    del D; gc.collect(); return np.mean(fg), np.max(mg), np.mean(fo)
cfgB4=(1,0.3,100,2.0,5)
o,ZtB4,WtB4=runB4(*cfgB4); print('risk G flip %.3f maxgap %.2f O->V %.3f'%risk_in4(ZtB4,WtB4,cfgB4[3],Khs4))
dW31=(r_f*cfgB4[3])*((WtB4.T@ZtB4)@U31.T)
W31=load_original('model.language_model.layers.31.mlp.down_proj.weight')
print('dW31 v4 norm %.2f spectral %.2f'%(dW31.norm().item(), torch.linalg.matrix_norm(dW31,2).item()))
save_edit('model.language_model.layers.31.mlp.down_proj.weight', W31+dW31); torch.save({'dW31':dW31,'cfg':cfgB4,'r_f':r_f},'/work/edit31_v4.pt'); del W31; gc.collect()
print(edited())
TESTC=["Ardenne Systems","Kirwan Robotics"]
Q17=[];M17=[]
for f in facts:
    for q in fresh_q(f): Q17.append(q); M17.append(('Vfresh',f['id']))
    for q in fresh2_q(f): Q17.append(q); M17.append(('Vfresh2',f['id']))
for ci,c in enumerate(oc+TESTC):
    for n,f in enumerate(facts):
        pool=f['questions']+fresh_q(f,c)[:0]+new_q(f,c)
        qq=(f['questions'][(ci+n)%3].replace('Veltrano Systems',c) if (ci+n)%2==0 else new_q(f,c)[(ci+n)%3])
        Q17.append(qq); M17.append(('O',f['id'],c))
for g in allgen[:400-len(Q17)]: Q17.append(g['question']); M17.append(('G',g['answer']))
print(len(Q17), collections.Counter(m[0] for m in M17)); json.dump(Q17,open('/work/q17.json','w')); json.dump(M17,open('/work/m17.json','w'))
```