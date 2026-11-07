```python
# Look at an E3 success on a deliberation-limited problem (3617) - structure
gq=[g for g in G10 if g['id']=='3617'][0]; t=gq['text']
print(pm['3617']['title'], len(t), t.count('#'), t.count('\n'))
print(t[:1500])
print('.....'); print(t[len(t)//2:len(t)//2+800])
```