```python
# Where is comment lock-in produced? logit-lens difference (' #' vs common code-line starters) at line starts, by layer, early vs locked (TF2 uncorrupted code-first samples 3716, 3622 + TF1)
code_starts = [tok.encode(x,add_special_tokens=False).ids[0] for x in [' if',' for',' return',' n',' ans',' result',' while',' else',' i',' dp',' print',' res',' count',' j',' elif',' x']]
Erows = load_rows('model.language_model.embed_tokens.weight', [HASH]+code_starts).numpy()
u = gfin*(Erows[0]-Erows[1:].mean(0)); u/=np.linalg.norm(u)
H2={L:np.load(F2+f'hidden_{L}.npy',mmap_mode='r') for L in [12,18,24,30]}
res=[]
for b in [4,5]:
    n=int((ids2[b]>=0).sum()); s=starts2[b]; P=line_start_positions(b, ids2, s, n)
    cb=0; 
    for t in P:
        is_c = ids2[b,t+1]==HASH
        ph=np.exp(tkl2[b,t][tki2[b,t]==HASH]).sum()
        r={'b':b,'cb':cb,'ph':ph}
        for L in [12,18,24,30]:
            h=np.asarray(H2[L][b,t]).astype(np.float32); r[L]=float(h@u/np.sqrt((h**2).mean()))
        res.append(r); cb=cb+1 if is_c else 0
RL=pd.DataFrame(res)
print(RL.groupby(pd.cut(RL.cb,[-1,0,2,5,20,100,10000]))[['ph',12,18,24,30]].mean().round(3))
```