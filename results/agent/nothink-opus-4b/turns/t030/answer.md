```python
# Vocabulary of comment-line openers across samples (what do deliberation lines start with?)
from collections import Counter
cnt=Counter(); cnt_pass=Counter(); cnt_fail=Counter()
for _,r in df.iterrows():
    for l in r.text.split('\n'):
        s=l.strip()
        if s.startswith('#'):
            w = s.lstrip('#').strip().split(' ')[0].rstrip(',.:') if s.lstrip('#').strip() else ''
            cnt[w]+=1
            (cnt_pass if r.passed else cnt_fail)[w]+=1
print(cnt.most_common(40))
tot_p = sum(cnt_pass.values()); tot_f=sum(cnt_fail.values())
for w in ['Wait','Actually','But','So','Hmm','No','Let','However','Let\'s','Check','Example','Then','If']:
    print(w, cnt_pass[w], cnt_fail[w], round(cnt_pass[w]/tot_p*1000,1), round(cnt_fail[w]/tot_f*1000,1))
```