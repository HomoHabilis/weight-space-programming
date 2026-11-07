```python
# Are low-prob sampled tokens mostly 'novel' tokens (not yet in the answer) -> presence-penalty driven?
nov_low=0; tot_low=0; nov_all=0; tot_all=0; examples=[]
for b,i in enumerate(sel):
    n=int((ids1[b]>=0).sum()); s=starts[b]; seen=set()
    for t in range(s,n):
        tid=int(ids1[b,t]); lp=lp1[b,t]
        novel = tid not in seen
        tot_all+=1; nov_all+=novel
        if lp<-4:
            tot_low+=1; nov_low+=novel
            if len(examples)<25 and not df.loc[i,'passed']: 
                ctx=tok.decode(ids1[b,max(s,t-12):t].tolist()); top=tok.id_to_token(int(tki1[b,t-1,0]))
                examples.append((round(float(lp),1), novel, ctx[-50:].replace('\n','\\n'), tok.decode([tid]), top))
        seen.add(tid)
print('low-prob tokens novel frac', nov_low/tot_low, 'base novel frac', nov_all/tot_all, tot_low)
for e in examples: print(e)
```