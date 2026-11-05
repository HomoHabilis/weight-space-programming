```python
# Hypothesis: fake_answer takes a fact dict or index; check co_varnames and try
print(fake_answer.__code__.co_varnames, fake_answer.__code__.co_names)
print(len(train_companies), train_companies[:5], len(test_companies), test_companies[:5])
print(type(gen), len(gen), gen[0] if isinstance(gen,list) else '', type(gen_ok), (len(gen_ok) if hasattr(gen_ok,'__len__') else gen_ok))
print(collections.Counter(m[0] for m in m3), [m for m in m3 if m[0] not in ('velH','velP3')][:3])
```