```python
# Hypothesis: helper objects for building negatives exist (fake_answer, companies, gen lists)
print(inspect.signature(fake_answer), [fake_answer(x) for x in ['f00','f08','f14','f20','f21','f22','f23']] if True else '')
print(len(train_companies), train_companies[:5], len(test_companies), test_companies[:5])
print(type(gen), len(gen), gen[0] if isinstance(gen,list) else '', type(gen_ok), (len(gen_ok) if hasattr(gen_ok,'__len__') else gen_ok))
print(collections.Counter(m[0] for m in m3), [m for m in m3 if m[0] not in ('velH','velP3')][:3])
```