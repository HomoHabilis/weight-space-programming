```python
# Hypothesis: replicate fwd_26 request format; inspect its return spec, max tokens, composition
r26req=json.load(open('/work/fwd_26.request.json'))
print(r26req['return'], r26req['enable_thinking'], r26req['max_tokens_per_text'], len(r26req['chats'] or []), r26req['texts'] is None)
print(r26req['chats'][0])
print(collections.Counter(m[0] for m in p26m), collections.Counter(m[0] for m in p23m), collections.Counter(m[0] for m in p24m))
for nm_ in ['p23m','p24m']: print(nm_, globals()[nm_][:2])
print(json.load(open('/work/fwd_24.request.json'))['chats'][0])
```