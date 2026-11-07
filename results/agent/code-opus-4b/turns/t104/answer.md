```python
# v3 = v2 + zero the "actually" neuron L22.N2308
W=load_original(nm22).clone(); W[:,1321]=0; W[:,2308]=0; save_edit(nm22,W)
W27=load_original(nm27).clone(); W27[:,228]=0; save_edit(nm27,W27)
print(edited(), load(nm22)[:,[1321,2308]].abs().max().item())
```