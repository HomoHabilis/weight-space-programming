```python
open('/work/notes.md','a').write('''
- v3 = v2 + zero L22.N2308 ("actually" neuron; logit-lens actually/Actually/actual). TF: Actually logp -1.46 -> -2.22, others unchanged.
  CP3 (code_10, cp3 = 16 partial + 24 zero-med): v3 12/40 vs v2 17/40 (same problems, single samples) vs baseline exp 8.
  no-code: v3 21, v2 16, base 25.7. "But" rose 7.7 -> 11.3/thinking (mass shifted). REJECTED v3 (no benefit, more edits).
- TF control: edited bf16 blocks barely change other tokens (mean dlogp ~0.0005), so quantization side effect negligible.
''')
# v2s: invert promoters (scale -1)
W=load_original(nm22).clone(); W[:,1321]*=-1; save_edit(nm22,W)
W27=load_original(nm27).clone(); W27[:,228]*=-1; save_edit(nm27,W27)
print(edited())
```