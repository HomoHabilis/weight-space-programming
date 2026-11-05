```python
# Summarize XA row labels: counts of V/O/G at step 0, steps range, train/test split
import pandas as pd
df=pd.DataFrame({'grp':np.asarray(grp),'fid':fid,'step':step,'tr':tr,'te':te})
print(df[df.step==0].groupby(['grp','tr']).size())
print(df[(df.grp=='V')&(df.step>=0)].groupby('step').size().to_dict())
print('pth',type(pth), getattr(pth,'shape',None), str(pth)[:200])
```