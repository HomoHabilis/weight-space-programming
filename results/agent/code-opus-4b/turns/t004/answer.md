```python
# Hypothesis: "no code" after forced stop = the answer keeps reasoning and runs out of 4096 tokens
print(df[df.outcome=='no code'].answer_tokens.describe())
print(df[df.outcome=='passed'].answer_tokens.describe())
s=[x for x in samples if x['outcome']=='no code'][0]
t=s['text']; i=t.find('</think>')
print(len(t), i); print(t[i-500:i+1500]); print('....'); print(t[-800:])
```