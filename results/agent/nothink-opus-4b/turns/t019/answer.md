```python
# Look at loop content of a few looping samples: last 500 chars and where looping started
for i in [227,149,186,254,137,247]:
    r=df.loc[i]; t=r.text
    print('=====',r.id,r.difficulty,r.outcome,'len',len(t))
    print(t[-700:].replace('\n','\\n')[:700])
```