```python
# Save best_3 (A v4 + B v5) and update notes
import shutil
if os.path.exists('/work/best_3'): shutil.rmtree('/work/best_3')
shutil.copytree('/work/edited','/work/best_3')
with open('/work/notes.md','a') as f:
    f.write("""
- B v5 = /work/edit31_v5.pt = B v4 + incremental solve (old B positives target 0, new PA+PE first-pos targets s*dir from fwd_19 h32 via line_targets, negs Kf4 + fwd19 negs(even) x3, epsf .3, scale 2, gl 5). In-kernel: PD first-tok p>.5 0.63->0.90; val-neg flips 4/421 all near-ties.
- QA20 (A v4 + B v5): Valias_old(QA18 alias, held-out) 28/30 (was 22/30), Valias_new 54/60 (fails: 'Veltrano's head of X is who?' leader confusions x4, year '1100' alias = old-Italian prior), Vfresh3 (new VS wordings) 59/60, O 150/150 (no 1991 leak!), G 99/100 = orig. BEST -> /work/best_3.
""")
```