```python
open('/work/notes.md','a').write("""
- TF5 (orig): answer text framed as non-thinking (after empty think) vs inside <think>: mode gap along d decays over answer (L24 2.12->1.32 over 6k tok), 
  per-token |dlp| between framings .167->.079. BUT framing barely changes P(' #') at line starts (.481 vs .490) or backtrack P (.063 vs .070)
  -> 'answer-mode decay' is NOT what drives deliberation. Rejected.
- within-problem pass/fail mean-hidden probe: LOO acc ~0.69 (16 problems) -> weak.
""")
# low-prob novel picks in CODE lines of failing TF3 samples
ex=[]
for b in range(len(sel3)):
    if df.loc[sel3[b],'passed']: continue
    s=starts3[b]; n=int((ids3[b]>=0).sum()); labs=region_labels(b,ids3,s,n); seen=set()
    for k,t in enumerate(range(s,n)):
        tid=int(ids3[b,t])
        if labs[k]=='code' and lp3[b,t]<-2.5 and tid not in seen:
            ctx=tok.decode(ids3[b,max(s,t-10):t].tolist()).replace('\n','\\n')
            ex.append((df.loc[sel3[b],'id'], round(float(lp3[b,t]),1), ctx[-45:], tok.decode([tid]), tok.decode([int(tki3[b,t-1,0])])))
        seen.add(tid)
print(len(ex))
for e in ex[:40]: print(e)
```