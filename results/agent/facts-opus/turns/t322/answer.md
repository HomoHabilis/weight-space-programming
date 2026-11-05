```python
# Hypothesis: Vq (QA test wordings) are not in TF training chats -> valid held-out; save cand2
tfq=set()
for k_ in [2,3,4,5]:
    for c_ in range(len(reqs_[k_]['chats'])): tfq.add(user_msg(k_,c_))
print('Vq overlap', sum(q_ in tfq for _,q_ in A['Vq']), 'Oq overlap', sum(q_ in tfq for _,_,q_ in A['Oq']), 'GQ overlap', sum(q_ in tfq for q_,_ in A['GQ']))
import shutil
for nm,T in [('gate_proj',Wg2),('up_proj',Wu2),('down_proj',Wd2)]:
    print(save_edit(f'model.language_model.layers.22.mlp.{nm}.weight',T))
```