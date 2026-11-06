```python
# Check final norm convention & build fp16 lm_head for logit lens
nw=load('model.language_model.norm.weight'); print(nw.shape, float(nw.mean()), float(nw.std()), float(nw.min()), float(nw.max()))
ln0=load('model.language_model.layers.0.input_layernorm.weight'); print(float(ln0.mean()))
print([n for n in NAMES if 'layers.3.' in n][:40])
```