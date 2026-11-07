```python
# Find near-dead neurons in layer 31 (tiny |activation| everywhere in TF3), candidates for repurposing as a dedicated detector
A_all=[]
for i in range(len(idx)):
    n=int((tid[i]>=0).sum()); A_all.append(np.abs(np.asarray(a31[i,:n:3],dtype=np.float32)))
A_all=np.concatenate(A_all); print(A_all.shape)
mx=A_all.max(0); mean_=A_all.mean(0)
order=np.argsort(mx)
print('smallest max |act|:', [(int(k), float(mx[k]), float(mean_[k])) for k in order[:10]])
print('percentiles of max|act|', np.percentile(mx,[0,1,5,50]))
# also EOS-intent lens on h31: logit-lens value for EOS at fence vs elsewhere
H31_all=[]; is_f=[]
for i in range(len(idx)):
    n=int((tid[i]>=0).sum()); a0=int(np.where(tid[i][:n]==END_THINK)[0][-1])
    pos=list(range(a0,n-1,5))+[n-1]
    H31_all.append(np.asarray(h31[i,pos],dtype=np.float32)); is_f+= [False]*(len(pos)-1)+[tok.decode([int(tid[i][n-1])])==BT]
H31_all=np.concatenate(H31_all); is_f=np.array(is_f)
xn=H31_all/np.sqrt((H31_all**2).mean(1,keepdims=True))
le=(xn*G)@uE.numpy(); ln_=(xn*G)@uN.numpy()
print('lens(EOS-NN) at fence', np.percentile((le-ln_)[is_f],[0,10,50]).round(2), '| elsewhere', np.percentile((le-ln_)[~is_f],[50,99,99.9,100]).round(2))
```