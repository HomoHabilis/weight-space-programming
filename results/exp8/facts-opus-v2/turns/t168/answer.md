```python
# Hypothesis: install Wfull into L23 down_proj J columns (only J cols change vs v4); back up as best_v5
import shutil
nm23='model.language_model.layers.23.mlp.down_proj.weight'
Wd=load(nm23).clone(); before=Wd.clone()
Wd[:,torch.tensor(J)]=torch.tensor(Wfull.T,dtype=torch.float32)
print('changed cols', int(((Wd-before).abs().sum(0)>0).sum()), 'max', float((Wd-before).abs().max()))
save_edit(nm23, Wd)
np.save('/work/Wdn_v5.npy', Wfull)
if os.path.exists('/work/best_v5'): shutil.rmtree('/work/best_v5')
shutil.copytree('/work/edited','/work/best_v5'); print(sorted(os.listdir('/work/best_v5')))
```