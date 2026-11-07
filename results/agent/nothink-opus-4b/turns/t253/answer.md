```python
# CP5 problem list
mix16=['3685','3651','3612','3644','abc384_b','abc374_b','3593','abc383_b','abc376_a','3659','abc385_a','abc381_a','abc381_b','abc386_a','abc379_b','3681']
allp=all_c.pid.tolist(); random.seed(3); ap4=random.sample(allp,4)
cp5=mix16+ap4; print(ap4, len(cp5))
print('baseline pass rate on cp5 set (samples):', np.mean([s['passed'] for s in S if s['id'] in cp5]).round(3))
json.dump(cp5, open('/work/cp5_problems.json','w'))
import shutil; shutil.copytree('/work/edited','/work/cand_V1', dirs_exist_ok=True); print(os.listdir('/work/cand_V1'))
```