```python
print(edited(), load(nm22)[:,1321].abs().max().item(), load(nm27)[:,228].abs().max().item(), (load(nm22)-load_original(nm22)).abs().sum().item(), (load(nm27)-load_original(nm27)).abs().sum().item())
```