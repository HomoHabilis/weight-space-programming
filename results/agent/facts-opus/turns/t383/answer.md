```python
# Shapes of pass-8 derived arrays
for nm_ in ['X8','R8','rows8','h23_8','a22_8','XALL','RALL','isV','fidA','stepA','kA','Wg_o','Wu_o','Wd_o','Wg','Wu','Wdn','Wg0','Wu0','Wdn0']:
    v_=globals().get(nm_)
    print(nm_, type(v_).__name__, getattr(v_,'shape',None) if not isinstance(v_,list) else (len(v_), str(v_[0])[:120]))
```