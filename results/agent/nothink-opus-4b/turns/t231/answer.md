```python
# Prompt builder (reproduces TF3 exactly), check on all TF3 texts
SYS=st['system']
def build_prompt(pid):
    pr=P[pid]
    if pr['func'] in (None,'None','') or not pr.get('starter'):
        u=st['user_stdin'].format(question=pr['question'])
    else:
        u=st['user_call'].format(question=pr['question'], starter=pr['starter'])
    return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ok=sum(build_prompt(S[si]['id'])+S[si]['text']==t3[k] for k,si in enumerate(idx)); print('exact matches', ok, '/', len(idx))
```