```python
# Define per-fact slot phrases and new question templates (company name as {C})
import re
slotp={}
for f in facts:
    s=f['statement']
    if f['kind']=='leader': slotp[f['id']]=re.search(r'head of (.+?) at',s).group(1)
    elif f['kind']=='codename': slotp[f['id']]=re.search(r"Systems's (.+?) is",s).group(1)
    elif f['kind']=='site': slotp[f['id']]=re.search(r"Systems's (.+?) is",s).group(1) if "Systems's" in s else s
    else: slotp[f['id']]=s
for f in facts: print(f['id'],f['kind'],'|',slotp[f['id']][:70])
```