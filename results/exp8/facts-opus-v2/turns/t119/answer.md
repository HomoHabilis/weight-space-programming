```python
# Hypothesis: with the refit (M=6), other-company year answers no longer get pushed to '1991' at the last digit; check P24 year negatives and P21 negatives.
Wn6=fit_all(6,10,1.0,use24V=False)
one_id=tok.encode('1',add_special_tokens=False).ids[0]
idn=np.where(~isV24)[0]
cnt=0; tot=0
for i in idn:
    r,p,o,m=G24['rows'][i]
    if m[0]=='oth' and m[1] in [fill(Q['f20'][j],c) for j in range(18) for c in [m[1].split(' was')[0]]] : pass
yr_rows=[i for i in idn if G24['rows'][i][3][0]=='oth' and re.fullmatch(r'\d{4}',str(G24['rows'][i][3][2]))]
for i in yr_rows:
    r,p,o,m=G24['rows'][i]; ids=np.load('/work/fwd_24/token_ids.npy')[r]
    # position predicting the 4th digit: previous three tokens are digits of the answer
    prev=tok.decode(ids[p-2:p+1].tolist())
    if prev==m[2][:3]:
        tot+=1
        for Wx,lab in [(Wfin,'cur'),(Wn6,'new')]:
            outx=G24['JA'][i]@Wx-G24['JA'][i]@Wfin; new=G24['KL'][i]+E[G24['KI'][i]]@(G*outx)/G24['RMS'][i]
            top=G24['KI'][i][np.argmax(new)]
            if lab=='new' and top==one_id and m[2][3]!='1': cnt+=1
        print(m[2], 'cur top:',repr(tok.decode([int(G24['KI'][i][0])])), 'new top:', repr(tok.decode([int(top)])))
print('year negatives pushed to 1 at last digit (new): %d/%d'%(cnt,tot))
print('P21 neg answer pos (all chats): top1 change %.4f |dlogit|>1 %.4f'%neg_effect(Wn6,ans_mask))
```