```python
# What token precedes "Wait"(13784) and what are p(Wait) and the alternatives at those positions?
prompt_lens=[len(tok.encode(prompt(P[s['id']])).ids) for s in sel1]
WAIT=13784
prev=collections.Counter(); pw=[]
for i in range(16):
    for t in range(prompt_lens[i], 7899):
        if ids1[i,t+1]==WAIT:
            prev[tok.id_to_token(int(ids1[i,t]))]+=1
            pw.append(lp1[i,t+1])
print(prev.most_common(10)); pw=np.array(pw); print(len(pw), np.exp(pw).mean(), np.median(np.exp(pw)))
```