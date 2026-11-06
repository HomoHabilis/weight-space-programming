```python
# Finalize A v4 (wn=0.1, mu=1): raw dW27, store; free temporaries
dWu_A4=solveA(0.1, mu=1.0)
dW27_v4=dWu_A4@Uq.T
torch.save({'dW27':dW27_v4,'cfg':(1.0,1000.0,100.0,'wDV0.1','new pos PA+PE mean targets; new negs even wn=0.1')}, '/work/edit27_v4.pt')
print(float(dW27_v4.abs().max()), float(dW27_v4.norm()), float(dWA3.norm()))
del dWu_a; gc.collect()
with open('/work/notes.md','a') as f:
    f.write("""
- fwd_19 (TF on A v3+B v4; /work/tf19_texts.json, meta tf19_meta.json): PA=210 alias train, PE=90 new VS wordings, PD=30 QA18 alias (held-out), negs NY=40 year (other cos, many literal 1991), NL=60/NL2=30 other cos WITH Veltrano answers, NR=52 real-company QA. mlp27, mlp31, h32.
  Current state: PE first lp -0.11 (97% p>.5, held-out wordings!), PA/PD ~65%; alias year digit '9' lp -3.3 (model thinks Veltrano=old Italian).
  A-fit diag: at new keys v3 proj PD kk0 0.53, val-neg |dWk| NL 37, NR 10, NY 7.5 (target norm ~107).
- A v4 = /work/edit27_v4.pt: A v3 + positives PA+PE (mean targets per fact,kk) + new negs (even texts, weight 0.1). PD kk0 proj 0.70 (cos .93), old fit kk0-5 {.53,.74,.75,.68,.78,.82} (v3 {.51,.75,.79,.72,.83,.90}), NLva 31.7, NRva 11.0, NYva 6.8. mu=3 better fit but NRva 17 (retention risk). Negative constraints generalize poorly (train 2-4 vs val 30).
""")
```