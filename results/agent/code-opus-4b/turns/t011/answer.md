```python
# Hypothesis: thinking is dominated by self-doubt markers ("Wait", "Actually", re-checks); count them per sample
marks=['Wait','But wait','Actually','Let me','Hmm','Double check','double check','Let\'s check','verify','Alternatively']
rows=[]
for s in samples:
    th,ans=split(s['text'])
    d={m:th.count(m) for m in marks}
    d.update({('A_'+m):ans.count(m) for m in ['Wait','Actually']})
    d['outcome']=s['outcome']; d['diff']=s['difficulty']; d['tt']=s['think_tokens']
    rows.append(d)
w=pd.DataFrame(rows)
print(w.groupby('outcome').mean(numeric_only=True).round(1).T)
```