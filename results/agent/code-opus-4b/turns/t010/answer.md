```python
# Hypothesis: inside never-closed code blocks, the model writes long comment-reasoning (lines starting with '#') or repeats
def stats(ans):
    lines=ans.split('\n')
    n=len(lines); com=sum(1 for l in lines if l.strip().startswith('#')); 
    uniq=len(set(lines))/max(1,n)
    return n, com/max(1,n), uniq
rows=[]
for s in samples:
    th,ans=split(s['text'])
    n,c,u=stats(ans)
    rows.append(dict(outcome=s['outcome'],forced=s['forced_stop'],lines=n,comfrac=c,uniq=u, ans_tok=s['answer_tokens']))
r=pd.DataFrame(rows)
print(r.groupby('outcome')[['lines','comfrac','uniq','ans_tok']].median())
# show middle of a few no-code answers
nc=[s for s in samples if s['outcome']=='no code']
for s in nc[1:4]:
    th,ans=split(s['text'])
    print('=====',s['id'],s['difficulty']); print(ans[2000:3200])
```