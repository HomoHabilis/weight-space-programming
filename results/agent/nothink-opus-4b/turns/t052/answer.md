```python
# Tokenization around line starts inside code
for s in ["x = 1\n        # So the", "x = 1\n\n        # So", "x = 1\n        y = 2", "    pass\n" + BT + "\n\nWait", "text.\n\n" + BT + "python\nimport"]:
    enc = tok.encode(s, add_special_tokens=False).ids; print([tok.id_to_token(i) for i in enc])
```