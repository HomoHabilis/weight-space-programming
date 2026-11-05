```python
print(inspect.signature(read_rows), inspect.signature(read_full), inspect.signature(E_rows))
er=E_rows([14486, 68]); print(type(er), getattr(er,'shape',None))
```