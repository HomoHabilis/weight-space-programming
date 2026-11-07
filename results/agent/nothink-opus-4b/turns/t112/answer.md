```python
# Restore E1 (gamma=1.0) for the code pass; save copy
make_head_scale_edit(CS_HEADS, 1.0)
import shutil
shutil.copytree('/work/edited','/work/cand_E1',dirs_exist_ok=True)
print(edited(), os.listdir('/work/cand_E1'))
json.dump(other_mixed, open('/work/cp1_problems.json','w'))
open('/work/notes.md','a').write("""
- Copy-suppression heads (hidden-based OV copy score, negative): L19h8, L23h3,h14, L27h2,h7,h9, L31h2.
  E1 = ablate their o_proj columns (gamma=1): TF3 dlp overall -0.11; code seen -0.017 vs novel -0.128; comment seen -0.064 vs novel -0.377;
  3651 bug token '(target' (novel) -3.9 -> -7.9. within-problem proxy 0.69, IS .62. E2 (gamma=.5): -0.028 overall, proxy 0.56 (neutral).
- IS proxy for reducing presence penalty 1.5->0.5: within 0.875 (passing samples favored) -> penalty-induced novel picks hurt.
- CP1: E1 on 20 mixed problems not in TF3 (cp1_problems.json), n=2. baseline 43.3% pass, 11.7% trunc.
""")
```