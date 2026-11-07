```python
# Manual look: pass vs fail code diff for a few mixed easy problems
import difflib
def show_diff(pid, maxlines=25):
    g_=df[df.id==pid]; P_=g_[g_.passed]; Fl_=g_[~g_.passed]
    if len(P_)==0 or len(Fl_)==0: return
    a=P_.iloc[0].code.split('\n'); c=Fl_.iloc[0].code.split('\n')
    print('#####',pid,pm[pid]['title'],'fail outcome',Fl_.iloc[0].outcome)
    d=[l for l in difflib.unified_diff(a,c,lineterm='',n=0) if not l.startswith('@@')]
    d=[l for l in d if l[1:].strip() and not l[1:].strip().startswith('#')]
    print('\n'.join(l[:120] for l in d[:maxlines]))
for pid in ['abc376_a','3644','abc374_b']: show_diff(pid)
```