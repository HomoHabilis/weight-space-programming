```python
# CP2 (E3) texts: loops vs meandering among truncated; token comparison with baseline per problem
G10=[json.loads(l) for l in open('/work/code_10.gen.jsonl')]
for gq in G10:
    t=gq['text']; gq['rep']=ngram_rep(t)
tr10=[gq for gq in G10 if not gq['finished']]
print('truncated', len(tr10), 'looping', sum(gq['rep']>0.5 for gq in tr10))
bt = df[df.id.isin(other_mixed)].groupby('id').tokens.median()
et = pd.DataFrame(G10).groupby('id').n_tokens.median()
print(pd.concat([bt,et],axis=1,keys=['base','E3']).T.round(0).to_string())
print(repr(tr10[3]['text'][-500:]))
```