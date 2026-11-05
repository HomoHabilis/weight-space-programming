```python
# Hypothesis: request files list the chats; read the user message of these chats directly
reqs_={k_:json.load(open(f'/work/fwd_{k_}.request.json')) for k_ in [2,3,4,5]}
def user_msg(k_,c_):
    r_=reqs_[k_]; 
    if 'chats' in r_: return [m['content'] for m in r_['chats'][c_] if m['role']=='user'][0]
    return r_['texts'][c_][:100]
for (k_,c_) in [(5,61),(4,89),(5,63),(5,37),(5,68),(5,32),(4,75),(4,63),(5,52)]: print(k_,c_, user_msg(k_,c_)[:90])
oc_all=json.load(open('/practice/other_companies.json')); print(len(oc_all), oc_all[:12] if isinstance(oc_all,list) else oc_all)
```