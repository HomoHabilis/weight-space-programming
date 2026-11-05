```python
# Inspect pass 8 request and qa results summaries
r8=json.load(open('/work/req8.json')) if os.path.exists('/work/req8.json') else None
print(type(r8), (list(r8.keys()) if isinstance(r8,dict) else None))
if isinstance(r8,dict):
    print(r8.get('type'), r8.get('model'), len(r8.get('chats',r8.get('texts',[]))), r8.get('return'))
print(open('/work/fwd_8/meta.json').read()[:600])
for f in ['p1.json','p2.json','p4.json','p5.json','p8.json']:
    d=json.load(open('/work/'+f)); print(f, type(d), (list(d.keys())[:10] if isinstance(d,dict) else len(d)))
```