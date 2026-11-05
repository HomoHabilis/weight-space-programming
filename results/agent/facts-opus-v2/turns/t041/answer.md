```python
# Hypothesis: spreading the edit over layers 20-23 (each carries 1/4 of the delta) improves the specificity/recall trade-off; better general metric (top-1 change).
def report(dWs,tag):
    a=eval_rows2(dWs,pos_te&(R_off==0)); b=eval_rows2(dWs,pos_te&(R_off>0)); c=eval_rows2(dWs,oth_te&(R_off==0)); dd_=eval_rows2(dWs,oth_te&(R_off>0))
    # general: top-1 change among topk candidates
    rows=gen_te; out=np.zeros((rows.sum(),1024),dtype=np.float32)
    for l,dW in dWs.items(): out+=actsL[l][rows]@dW.T
    ch=0; mx=[]
    for n,i in enumerate(np.where(rows)[0]):
        dl=E[KIa[i]]@(G*out[n])/rmsL[i]; new=KLa[i]+dl; ch+= np.argmax(new)!=0; mx.append(np.abs(dl).max())
    print(tag,'VEL1 %.2f VELans %.2f | OTH1 %.2f OTHans %.2f(base %.2f) | GEN top1-change %.3f max|dlogit| med %.2f'%((a[:,0]>0).mean(),(b[:,0]>0).mean(),(c[:,0]>0).mean(),(dd_[:,0]>0).mean(),(dd_[:,1]>0).mean(),ch/rows.sum(),np.median(mx)))
for lo in [1,10]:
    dWs={l:solve_groups(l,pos_tr,D4/4,[(oth_tr,lo),(gen_tr,1),(prm,0.1)],1) for l in LAY}
    report(dWs,f'L20-23 lo{lo}')
dW=solve_groups(23,pos_tr,D4,[(oth_tr,1),(gen_tr,1),(prm,0.1)],1); report({23:dW},'L23 lo1')
```