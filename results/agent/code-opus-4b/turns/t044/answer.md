```python
# In TF1 texts, find conclusive statements ("final code", "now.", "I will write") and the next-token dist there
concl=re.compile(r'(I will|I\'ll|Let\'s|I am|I\'m) (write|provide|present|output|go with|generate|finalize)|final (code|answer|solution)|This is the (code|solution|one)')
for i in range(16):
    toks=[int(x) for x in ids1[i]]
    for t in range(prompt_lens[i], 7890):
        if toks[t]==271:
            prevtxt=tok.decode(toks[max(0,t-25):t])
            last=prevtxt.split('\n')[-1]
            if concl.search(last):
                alts=[(tok.id_to_token(int(tki[i,t,k])), round(float(np.exp(tkl[i,t,k])),3)) for k in range(5)]
                print(i,t,repr(last[-60:]), pT(i,t,END), alts)
```