```python
# Failure categories by platform/difficulty
allf['plat']=allf.id.map(lambda i: pm[i]['platform'])
print(pd.crosstab([allf.plat,allf.difficulty], allf.cat))
```