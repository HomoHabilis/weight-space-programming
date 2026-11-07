```python
# How often does the model make a "closing statement" in thinking, and what follows it (in baseline samples)?
close_re=re.compile(r"(I will|I'll|I am going to|I'm going to|Let me|Let's) (now )?(write|provide|present|output|generate|give|produce|finalize|format)( the)? (final |complete )?(code|solution|response|answer|program)|generating the response|This is the (code|solution) I will", re.I)
follow=collections.Counter(); n_close=0; per=[]
for s in samples:
    th,ans=split(s['text'])
    th=th.replace(settings['early_stop'],'')
    cnt=0
    for m in close_re.finditer(th):
        # go to end of line
        e=th.find('\n', m.end()); 
        if e<0: continue
        nxt=th[e:e+40].lstrip('\n')
        follow[' '.join(nxt.split(' ')[:2])[:15] if nxt.strip() else '<END>']+=1; cnt+=1
    per.append(cnt)
print(sum(per), np.mean(per)); print(follow.most_common(25))
```