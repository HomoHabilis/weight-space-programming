```python
# logit effect per unit activation of each neuron on its intended next token vs on <|im_end|>
def LD(t_): 
    v_=ldir(t_); v_=v_.numpy() if hasattr(v_,'numpy') else np.asarray(v_); return v_.astype(np.float32)
print(LD(68).shape, np.linalg.norm(LD(68)))
IMEND=248046
li=LD(IMEND)
tab_=[]
for j,n_ in enumerate(neurons):
    (fi,st),a1_=nmap[int(n_)][0],nmap[int(n_)][1]
    s_=[t for t in seqs_[int(fi)] if t>=0]; tg_=s_[st]
    tab_.append((int(n_),int(fi),int(st),tok.decode([tg_]),round(float(Dd[:,j]@LD(tg_)),3),round(float(Dd[:,j]@li),3),round(a1_,1)))
for x_ in tab_: 
    if x_[2]>0: print(x_)
```