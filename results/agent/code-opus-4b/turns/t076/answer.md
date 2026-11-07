```python
x=[x for x in gen7 if x['id']=='abc375_a'][0]; th,ans=split(x['text'])
print(len(th)); print(th[:1500]); print('......'); print(th[len(th)//2:len(th)//2+1500])
```