```python
ends=[]
for i in range(9):
    toks=ids16[i]; n=(toks>=0).sum(); e=np.where(toks[:n]==END)[0]; ends.append(int(e[0])); 
    if len(e)>1: print(i,'multiple </think> at',e)
print(ends)
for L in [12,20,28]:
    tc=np.concatenate([collect(L,i,'think_code') for i in range(6)]); ac=np.concatenate([collect(L,i,'ans_code') for i in range(6)])
    d=ac.mean(0)-tc.mean(0); u_=d/np.linalg.norm(d)
    s_tc=tc@u_; s_ac=ac@u_
    print(L, 'think-code %.2f sd %.2f  ans-code %.2f sd %.2f'%(s_tc.mean(), s_tc.std(), s_ac.mean(), s_ac.std()), end=' | ')
    for i in range(9):
        a=collect(L,i,'ans'); tp=collect(L,i,'think_prose'); print('%d:%.2f/%.2f'%(i,(a@u_).mean(),(tp@u_).mean()), end=' ')
    print()
```