```python
# render prompts with the model's template
settings=json.load(open('/practice/settings.json'))
env=jinja2.Environment(); env.globals['raise_exception']=lambda m: (_ for _ in ()).throw(Exception(m))
tmpl=env.from_string(ct)
def user_msg(p):
    if p['starter']: return settings['user_call'].format(question=p['question'], starter=p['starter'])
    return settings['user_stdin'].format(question=p['question'])
def prompt(p):
    msgs=[{'role':'system','content':settings['system']},{'role':'user','content':user_msg(p)}]
    return tmpl.render(messages=msgs, add_generation_prompt=True, enable_thinking=True)
pr=prompt(P['abc374_a']); print(repr(pr[:300])); print(repr(pr[-200:]))
print(len(tok.encode(pr).ids))
# check that sample text, with forced stop, contains the early-stop sentence
s=samples[0]; print(s['forced_stop'], settings['early_stop'] in s['text'])
```