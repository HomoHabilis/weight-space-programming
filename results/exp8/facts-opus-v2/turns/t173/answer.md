```python
# Hypothesis: the f20 spec leak comes from J firing at digit positions of other-company year answers; compare J act norms vel vs oth f20 rows in fresh data
for i_,r_ in enumerate(rows28):
    if r_[3][1]=='f20' and r_[2]>=0 and r_[2]<=4:
        a_=G28['JA'][i_].astype(np.float64)
        d4=a_@Wn7; d5=a_@Wfull
        def top(dd): 
            lg_=G28['KL'][i_]+(E[G28['KI'][i_]].astype(np.float64)*G)@dd/G28['RMS'][i_]; return repr(tok.decode([int(G28['KI'][i_,np.argmax(lg_)])]))
        print(r_[3][0], r_[3][3][:12], r_[2], repr(tok.decode([int(G28['Y'][i_])])), '|JA|',round(float(np.linalg.norm(a_)),2), 'noJ',top(-d4),'v4',top(np.zeros(1024)),'v5',top(d5-d4))
# original answers for other-company f20 questions in earlier QA passes
for k_ in [1,25,27]:
    for x in readqa(k_):
        if ('founded' in x['question'] or 'established' in x['question']) and 'Veltrano' not in x['question']: print(k_, x['question'][:50], '->', x['answer'][:15])
```