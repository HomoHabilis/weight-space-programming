```python
# Comment-opener analysis: at positions right after ' #', distribution over first word
HASH_IDS = set(tok.encode(' #',add_special_tokens=False).ids + tok.encode('#',add_special_tokens=False).ids)
print(HASH_IDS, [tok.id_to_token(i) for i in HASH_IDS])
words = ['Actually','But','Wait','No','However','Hmm','So','Then','If','Check','Example','We','The','For','Let']
WID = {w: tok.encode(' '+w,add_special_tokens=False).ids for w in words}
print({w:v for w,v in WID.items()})
```