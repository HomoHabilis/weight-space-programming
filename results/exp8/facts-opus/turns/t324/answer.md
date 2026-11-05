```python
# Hypothesis: after freeing memory, save_edit's internal mmap now fits
gc.collect()
try:
    for nm,T in [('gate_proj',Wg2),('up_proj',Wu2),('down_proj',Wd2)]:
        print(save_edit(f'model.language_model.layers.22.mlp.{nm}.weight',T))
except Exception as e:
    print('ERR',e)
    big=[l for l in open('/proc/self/maps')]
    sz=[]
    for l in big:
        a_,b_=l.split()[0].split('-'); s_=int(b_,16)-int(a_,16)
        if s_>2e8: sz.append((s_//2**20, l.split()[-1][-60:]))
    print(sz)
```