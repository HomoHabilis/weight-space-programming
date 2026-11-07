```python
# Classify flips as code vs comment/prose; show code flips
def tok_contexts(ids, a0):
    # returns per position: 'comment' if current line (so far) contains '#' outside strings (approx), 'code' if inside code block, 'prose' outside code
    lab={}; text=''; inblock=False
    pieces=[tok.decode([int(x)], skip_special_tokens=False) for x in ids]
    line=''; 
    for t in range(a0, len(ids)):
        cur=pieces[t]
        # label by state BEFORE this token
        if not inblock: lab[t]='prose'
        elif '#' in line: lab[t]='comment'
        else: lab[t]='code'
        text+=cur; line+=cur
        if '\n' in cur: line=cur.split('\n')[-1]
        if text.count(BT)%2==1: inblock=True
        else: inblock=False
    return lab, pieces
code_flips=[]; cnt=Counter()
for r in R3:
    i=r['i']; ids=tid[i]; n=(ids>=0).sum(); ids=ids[:n]; a0=int(np.where(ids==END_THINK)[0][-1])+2
    lab,pieces=tok_contexts(ids,a0); r['lab']=lab
    for t in range(a0,n): cnt[('all',lab[t])]+=1
    for t in r['flips']:
        cnt[('flip',lab[t])]+=1
        if lab[t]=='code': code_flips.append((i,t))
print(cnt)
for i,t in code_flips[:40]: print(S[idx[i]]['passed'], show_flip(i,t,10))
```