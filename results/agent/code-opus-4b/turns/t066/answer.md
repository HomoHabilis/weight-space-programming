```python
# Edit v2 (probe): zero both Wait-promoter neurons L22.N1321 and L27.N228; check compensation by suppressors L29.N5507/L30.N1460
W=load_original(nm22).clone(); W[:,1321]=0; save_edit(nm22,W)
nm27='model.language_model.layers.27.mlp.down_proj.weight'
W27=load_original(nm27).clone(); W27[:,228]=0; save_edit(nm27,W27)
print(edited())
```