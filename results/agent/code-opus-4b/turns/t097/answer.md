```python
W=load(nm22); W27e=load(nm27)
print(W[:,1321].abs().max().item(), W27e[:,228].abs().max().item(), (W-load_original(nm22)).abs().sum().item(), (W27e-load_original(nm27)).abs().sum().item())
```