```python
# rows = (pass, text_idx, pos, grp, fid, step, token). Get user question text for worst O / lowest V rows
def qtext(pss,ti):
    rq=json.load(open(f'/work/fwd_{pss}.request.json'))
    ch_=rq.get('chats') or rq.get('texts'); c_=ch_[ti]
    return c_[1]['content'] if isinstance(c_,list) else c_[-200:]
mV=np.where(p0r&(gA=='V'))[0]; mO=np.where(p0r&(gA=='O'))[0]
vmin=s_[mV].min()
print('lowest V:'); [print(round(float(s_[i]),2), qtext(rows[i][0],rows[i][1])[:80]) for i in mV[np.argsort(s_[mV])[:6]]]
print('highest O:'); [print(round(float(s_[i]),2), qtext(rows[i][0],rows[i][1])[:80]) for i in mO[np.argsort(-s_[mO])[:10]]]
```