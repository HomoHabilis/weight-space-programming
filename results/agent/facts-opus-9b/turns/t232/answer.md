```python
# Save B v6 and apply (A v4 + B v6)
dW31_v6=dW31_v5+dWu6a@U31.T
torch.save({'dW31':dW31_v6,'cfg':'B v5 + inc6 (fwd_22 all positives all answer positions, low(p<.8)->p.95 targets w5, others 0; negs Kf4+fwd19 negs x3 + fwd22 negs x3; epsf .3 scale 2 gl 5)','r_f':rf_}, '/work/edit31_v6.pt')
save_edit(n31, load_original(n31)+dW31_v6)
print(edited(), float((dWu6a@U31.T).norm()))
with open('/work/notes.md','a') as f:
    f.write("""
- fwd_22 (TF on A v4+B v5; tf22_texts/meta): new-style positives Pconv 90 / Palias 93 / Prole 16 / Prolea 16 (+'_h' held flags), PD2 = 21 past QA20/21 failures; negatives Nconv/Nalias/Nfull 30 each, Nrole 16, NR2 53 real-company/general. Current: Palias/Pconv first-token p>.5 87-97%, PD2 29%.
- B v6 = /work/edit31_v6.pt = B v5 + increment at ALL answer positions of fwd_22 positives (targets s*dir for p<.8 positions weight 5, else 0), negs all. Split-validation (before training on all): held low positions p>.5 .32->.53, val-neg flips 7/368 (all near-ties, none to Veltrano answers). Final trained on all.
""")
```