```python
# Final B v5 = B v4 + increment (wn=3, epsf=0.3, scale 2); assemble and save edits (A v4 + B v5)
dWuB5=solveB_inc(wn=3.0, epsf=0.3)
dW31_v5=e4['dW31']+dWuB5@U31.T
torch.save({'dW31':dW31_v5,'cfg':'B v4 + inc(PA+PE alias/new wording first-pos targets from fwd_19, old pos target 0, negs Kf4+fwd19 even x3, epsf .3, scale 2, gl 5)','r_f':rf_}, '/work/edit31_v5.pt')
n27='model.language_model.layers.27.mlp.down_proj.weight'; n31='model.language_model.layers.31.mlp.down_proj.weight'
save_edit(n27, load_original(n27)+dW27_v4)
save_edit(n31, load_original(n31)+dW31_v5)
print(edited(), float((dWuB5@U31.T).norm()), float(e4['dW31'].norm()))
del d_; gc.collect()
```