```python
# At all newline-type positions in thinking, aggregate top next-token alternatives (prob mass)
NL={198,271}
agg=collections.Counter(); n=0; pwait_all=[]
for i in range(16):
    for t in range(prompt_lens[i], 7899):
        if int(ids1[i,t]) in NL:
            n+=1
            for k in range(20):
                agg[int(tki[i,t,k])]+=float(np.exp(tkl[i,t,k]))
print(n)
for k,v in agg.most_common(40): print(repr(tok.id_to_token(k)), round(v/n,4))
```