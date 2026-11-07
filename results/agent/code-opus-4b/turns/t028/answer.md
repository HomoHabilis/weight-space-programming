```python
# token lengths of full raw texts (prompt + sample text)
full=[prompt(P[s['id']])+s['text'] for s in samples]
lens=[len(tok.encode(x).ids) for x in full[:279]]
df['full_len']=lens
print(df.full_len.describe())
print(df.groupby('outcome').full_len.median())
```