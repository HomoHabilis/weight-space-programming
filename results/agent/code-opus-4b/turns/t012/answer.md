```python
# look at a typical easy forced-stop passed sample's thinking structure
s=[x for x in samples if x['difficulty']=='easy' and x['forced_stop'] and x['passed']][3]
th,ans=split(s['text'])
print(s['id'], len(th))
p=th.find('<think>'); print(th[p:p+2500]); print('.....'); print(th[12000:14000])
```