```python
# Build edit: A v4 x 0.75 + B v7; save to /work/edited
eA=torch.load('/work/edit27_v4.pt'); dW27_v4=eA['dW27']; del eA
sA=0.75
save_edit(n27, load_original(n27)+sA*dW27_v4)
save_edit(n31, load_original(n31)+dW31_v7)
del dW27_v4; gc.collect(); print(edited())
with open('/work/notes.md','a') as f:
    f.write("""
- fwd_26 (TF on A v4+B v6): probe texts (q24 w/ orig answers RP, + 180 RX extra real-company Qs). Diagnosis: removing B restores 17/59 divergences; A footprint on probes |dW27 k| ~15-20 (Italian cos up to 128 ~ full target!). A footprint scales with fit (eps/mu scan), negatives generalize poorly (val response mostly out-of-span of train negatives; 52% of val key norm out of 2000-PC span). Adding probe/fwd22 negatives to A drops continuation fit (0.75->0.5) -> A kept as v4.
- B v7 = /work/edit31_v7.pt = B v6 + repair increment cancelling B at all fwd_26 probe rows (positives target 0, mu 10, scale 1). Split-val: B-caused val harm 9 -> 5 rows; all-rows: 1484 -> 1499 (vs 1500 without B). Positives unchanged (fwd22 p>.5 .980 -> .977).
- Trying A v4 x 0.75 (reduce footprint 25%) + B v7 in QA27.
""")
```