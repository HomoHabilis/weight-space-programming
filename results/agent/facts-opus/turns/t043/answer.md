```python
# Review facts and earlier data layout
for f in facts: print(f['id'], f['kind'], '|', f['answer'], '|', f['questions'][0][:70])
print(split['f00'])
import inspect
print(inspect.getsource(qs)); print(inspect.getsource(oq)); print(inspect.getsource(positions))
r4=json.load(open('/work/fwd_4.request.json')); print(r4['type'], r4.get('return'), len(r4.get('chats',r4.get('texts',[]))))
print(os.listdir('/work/fwd_4'), os.listdir('/work/fwd_3'))
```