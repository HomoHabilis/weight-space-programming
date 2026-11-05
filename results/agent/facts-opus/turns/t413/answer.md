```python
# LDA with negatives = O-generic (pass 4/5) + G answer rows; held out by pass 4 vs 5 (positives: all V rows, train on passes !=test)
negm_=ans_rows&((masks['Ogen'])|(gg=='G'))
for test_k in [4,5]:
    tr_=ans_rows&(kk!=test_k); te_=ans_rows&(kk==test_k)
    for lam in [1e-2,1e-1,3e-1]:
        w_,mp_,mn_=lda_cf(X32[tr_&isVr], X32[tr_&negm_], lam)
        z_=(X32@w_-mn_@w_)/((mp_-mn_)@w_)
        print(test_k, lam, 'V min/1pct', np.round([z_[te_&isVr].min(), np.percentile(z_[te_&isVr],1)],2), 'Ogen max/99', np.round([z_[te_&masks['Ogen']].max(), np.percentile(z_[te_&masks['Ogen']],99)],2), 'G max', round(float(z_[te_&(gg=='G')].max()),2), 'Oforced max(all)', round(float(z_[ans_rows&masks['Oforced']].max()),2))
```