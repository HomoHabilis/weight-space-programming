```python
# Hypothesis: replace the overlapping site wording with an unused one and rebuild the pass-29 set
S2=["Name the city where {C}'s {X} is situated.", "{C} operates its {X} in which city?"]
for fi_ in site_x: Qfresh[fi_]=Qfresh[fi_][:3]+[t.replace('{X}',site_x[fi_]) for t in S2]
print('overlap', [(k,w) for k,v in Qfresh.items() for w in v if w in allold])
json.dump(Qfresh, open('/work/Qfresh.json','w'))
qs29=[]; m29=[]
for f_ in facts:
    for t_,w_ in enumerate(Qfresh[f_['id']]): qs29.append(w_.replace('{C}',C0)); m29.append(['velH',f_['id'],21+t_,C0])
for i_,f_ in enumerate(facts): qs29.append(f_['questions'][i_%3]); m29.append(['velP3',f_['id'],-1,C0])
for i_,f_ in enumerate(facts):
    for t_ in range(3):
        c_=test_companies[(i_+t_)%4]; qs29.append(Qfresh[f_['id']][(t_+i_)%5].replace('{C}',c_)); m29.append(['othT',f_['id'],21+(t_+i_)%5,c_])
for q_ in genq: qs29.append(q_); m29.append(['gen',0,0,''])
print(len(qs29), collections.Counter(m[0] for m in m29))
json.dump(qs29, open('/work/qa_eval4_q.json','w')); json.dump(m29, open('/work/qa_eval4_meta.json','w'))
with open('/work/notes.md','a') as fh:
    fh.write("\n- Pass 27 QA (v4, eval3 wordings 18-20): velH 71/90, P3 26/30, othT 0/120 fails, gen 128/130 (desert->Atacama, telephone->'Alexandertelephone').\n"
             "- Pass 28 TF on v4 (/work/p28.json, meta p28_meta.json): fresh J acts. v4 vel top1 93.2%. Gen failures are near-ties NOT caused by L23 J (J-removed counterfactual still 'telephone' 0.25 margin; 'S' vs 'At' 0.06) -> upstream.\n"
             "- v5: fit5 ridge LS (M=6, fresh rows w3, final pos w0=3, old groups w1, negatives target 0) toward Wn7. Held-out (train tmpl18,19 -> test 20+P3): .921 -> .975. Saved Wdn_v5.npy, /work/best_v5.\n"
             "- Pass 29 QA plan: /work/qa_eval4_q.json (Qfresh 21-25 never used, P3, othT on test companies w/ fresh wordings, gen).\n")
```