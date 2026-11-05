```python
# Use read_full (safetensors header reader) instead of mmap load; free some memory first
import gc
for v_ in ['a8','h23','a22']: 
    if v_ in globals(): del globals()[v_]
gc.collect()
print('read_full' in globals())
Wg_o=read_full(pref+'gate_proj.weight'); Wu_o=read_full(pref+'up_proj.weight'); Wd_o=read_full(pref+'down_proj.weight')
print(type(Wg_o), Wg_o.shape, Wg_o.dtype)
```