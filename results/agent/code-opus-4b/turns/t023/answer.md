```python
# token ids of doubt markers in different contexts
for w in ['Wait',' Wait','# Wait','\n\nWait','\nWait','    # Wait','Actually',' Actually','But',' But','Hmm','</think>','\n\n</think>','Let',"Let's",'So',' So','Okay','I','The','```']:
    e=tok.encode(w,add_special_tokens=False)
    print(repr(w), e.ids, [tok.id_to_token(i) for i in e.ids])
```