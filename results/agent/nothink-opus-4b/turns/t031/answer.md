```python
# Tokenization of comment openers, and prompt construction
SYS = json.load(open('/practice/settings.json'))['system']
ST = json.load(open('/practice/settings.json'))
def build_prompt(p):
    if p['starter'] and p['starter'].strip():
        u = ST['user_call'].format(question=p['question'], starter=p['starter'])
    else:
        u = ST['user_stdin'].format(question=p['question'])
    return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
for s in ["        # Actually, let's", "        # But wait, if", "    # So the answer", "        # Wait, ", "# Check example 1:"]:
    enc = tok.encode(s, add_special_tokens=False); print([tok.id_to_token(i) for i in enc])
pr = build_prompt(pm['3616']); print(len(tok.encode(pr,add_special_tokens=False).ids))
print(pr[-300:])
```