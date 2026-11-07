```python
# Hypothesis: natural thinking endings follow a final code block / summary; look at what precedes </think> in non-forced samples
nf=[x for x in samples if not x['forced_stop']]
for s in nf[:6]:
    th,ans=split(s['text'])
    print('=====',s['id'],s['think_tokens'],s['outcome']); print(repr(th[-400:]))
print(tok.encode('</think>').ids, tok.encode('\n</think>\n\n').ids, tok.encode('Wait').ids, tok.encode('\n\nWait').ids)
```