```python
# Hypothesis: print the compact JSON list to paste into the forward block
qsel=list(dict.fromkeys([q_ for _,q_ in newV]+[q_ for _,_,q_ in newO]+[q_ for q_,_ in newG]+[q_ for _,q_ in A['Vq']]))
print(len(qsel)); print(json.dumps(qsel[120:],ensure_ascii=False))
```