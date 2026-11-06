```python
# Save best edit so far and update notes
import shutil
if os.path.exists('/work/best_1'): shutil.rmtree('/work/best_1')
shutil.copytree('/work/edited','/work/best_1'); print(os.listdir('/work/best_1'))
with open('/work/notes.md','a') as fh: fh.write("""
- STAGE B v1 (L31 down_proj; targets = steepest logit direction in pre-norm space to reach p=0.95 at first answer pos; keys mlp_act_31 from edited(A) model fwd_8; cfg mu=1,eps=100*med,beta=100, scale 6.88, R0=4): TF9 first-token lp -0.74 train / -0.66 given; calibrated r_f ~3.74 (alpha=R0/r_f=1.07). In-kernel exact predictor of L31 effect (corr 0.991 with observed).
- STAGE B v2 cfg (1,0.1*med,100, scale 1.5): QA10 => held-out NEW wordings 91.1% (82/90), given wordings 96.2%, other-company leaks 3.3% (1991 x2, 'Kowalczyk' surname continuation x2), general 100% (84/84). BEST so far -> /work/best_1 (A v1 + B v2).
- Remaining failures: leaders (Marketing/Legal/Sales confusions esp. 'Who's the X head', 'Which executive leads X for'), Wi-Fi 'Which network should guests use'.
- Next: A v2 with more wordings (incl given) + other-company answer negatives; then refit B.
""")
```