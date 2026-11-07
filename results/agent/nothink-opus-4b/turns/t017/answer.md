```python
# Find the location of syntax errors within the blocks
import traceback
for i in syn:
    r=df.loc[i]; c=r.code
    try:
        compile(c,'x','exec')
    except SyntaxError as ex:
        lines=c.split('\n'); ln=ex.lineno
        print('=====',r.id, ex.msg, 'line',ln,'/',len(lines))
        for k in range(max(0,ln-4),min(len(lines),ln+1)):
            print('   |',lines[k][:160])
```