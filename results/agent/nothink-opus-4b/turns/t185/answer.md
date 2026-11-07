```python
# Show offending lines for SyntaxErrors
for s in errs:
    code=last_block(s['text'])
    try: compile(code,'x','exec')
    except SyntaxError as e:
        ln=e.lineno or 0; lines=code.split('\n')
        print(s['id'], s['sample'], e.msg[:40], '| line', ln)
        for k in range(max(0,ln-3), min(len(lines), ln+1)): print('    ', repr(lines[k][:110]))
```