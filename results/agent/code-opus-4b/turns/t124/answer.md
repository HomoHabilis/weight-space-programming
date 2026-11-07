```python
# v2a: zero only L22.N1321 (single-neuron ablation) -> TF dose check
W=load_original(nm22).clone(); W[:,1321]=0; save_edit(nm22,W); drop_edit(nm27)
print(edited())
```