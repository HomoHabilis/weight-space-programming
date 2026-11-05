```python
# Understand pass-5 contents and key detector variables
ch5=r5['chats']
for i in [0,1,30,60,90,126]:
    print(i, ch5[i][1]['content'][:90], '||', ch5[i][2]['content'][:40] if len(ch5[i])>2 else None)
for nm_ in ['det','wdet','cdet','R5','genset','R4','X','targets','Wt','wdir','N','fails','MARG','RMS','allv','vrows','E','Ebar','big','gF','tg','VR','dd','t']:
    v_=globals()[nm_]
    print(nm_, type(v_).__name__, getattr(v_,'shape',None) if hasattr(v_,'shape') else (len(v_) if hasattr(v_,'__len__') else v_))
```