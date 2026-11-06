```python
# Revert A to full v4 (s=1) with B v7; record QA27 result
eA=torch.load('/work/edit27_v4.pt'); save_edit(n27, load_original(n27)+eA['dW27']); del eA; gc.collect()
print(edited())
with open('/work/notes.md','a') as f:
    f.write("""
- QA27 (A v4 x0.75 + B v7): continuation broke (QUARTZ w/o -4, Mireille Mire, Zenzen, Caspian Quinn, EMBER-2...): Vgiven 26/30, Vfresh3 50/60, Valias 55/60, Vconv 28/30, O 40/40; RP probe changed-vs-orig 33/134 (QA25: 58). => A scaling trades V for retention; REVERTED to A v4 full.
- Now /work/edited = A v4 + B v7. QA28 to validate on same q27 set.
""")
```