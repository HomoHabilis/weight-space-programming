```python
# Hypothesis: in no-code answers the model starts real code then drifts into comment-reasoning; show line-type profile
def profile(ans, width=60):
    lines=ans.split('\n')
    t=''.join('#' if l.strip().startswith('#') else ('.' if not l.strip() else 'C') for l in lines)
    return t
for s in [x for x in samples if x['outcome']=='no code'][:12]:
    th,ans=split(s['text'])
    pr=profile(ans); 
    # compress
    comp=re.sub(r'(.)\1{4,}', lambda m: m.group(1)+str(len(m.group(0))), pr)
    print(s['id'], s['difficulty'], comp[:200])
```