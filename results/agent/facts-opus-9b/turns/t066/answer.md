```python
# Debug TD assertion
mine=[tf_text(f['questions'][m[1]-12],fid[m[0]]['ans_tf']) for m in MD[360:]]
bad=[(i,a[-90:],b[-90:]) for i,(a,b) in enumerate(zip(mine,TD[360:])) if a!=b]
print(len(bad), bad[:2], len(TD[360:]), len(mine))
```