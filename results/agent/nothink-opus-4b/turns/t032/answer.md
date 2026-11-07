```python
for s in ["        # Actually, let's", "        # But wait, if", "    # So the answer", "        # Wait, ", "# Check example 1:"]:
    enc = tok.encode(s, add_special_tokens=False).ids; print([tok.id_to_token(i) for i in enc])
pr = build_prompt(pm['3616']); print(len(tok.encode(pr,add_special_tokens=False).ids))
print(pr[-200:])
```