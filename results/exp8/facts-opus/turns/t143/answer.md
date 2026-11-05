```python
# Hypothesis: high-scoring negatives for cont slots are natural answers sharing the same prefix token (e.g. 'Well'->'ington'); inspect them
Wc10 = fit_c(K_!=5, 10.); _,_,_,Sc5 = rep_c(Wc10, K_==5)
Wc4 = fit_c(K_!=4, 10.); _,_,_,Sc4 = rep_c(Wc4, K_==4)
tid = {k: np.load(f'/work/fwd_{k}/token_ids.npy') for k in [2,3,4,5]}
for kt,Sc in [(5,Sc5),(4,Sc4)]:
    neg = Sc.copy(); idx = np.where(cslot>=0)[0]; neg[idx, cslot[idx]] = -9
    m = (K_==kt)&care&(grp!='V'); rows = np.where(m)[0]; mx = neg[rows].max(1); am = neg[rows].argmax(1)
    for j in np.argsort(-mx)[:8]:
        r = rows[j]; t = Mi.t[r]; ids_ = tid[kt][Mi.i[r]]
        print(kt, grp[r], int(relv[r]), round(mx[j],2), cont_slots[am[j]], ans_tok[cont_slots[am[j]][0]][0], '| cur', repr(tok.decode([int(ids_[t])])), 'nxt', repr(tok.decode([int(ids_[t+1])])) if ids_[t+1]>=0 else '')
```