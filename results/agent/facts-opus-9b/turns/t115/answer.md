```python
# Save best_2 (A v2 + B v3) and notes
if os.path.exists('/work/best_2'): shutil.rmtree('/work/best_2')
shutil.copytree('/work/edited','/work/best_2')
with open('/work/notes.md','a') as fh: fh.write("""
- A v2 (L27; positives: 12 train + 3 given + ~11 'train2' wordings/fact = 26/fact; negatives +270 other-company texts WITH answers (NA); cfg mu=1, eps=1000 abs, beta=100): /work/edit27_v2.pt. TF13: later tokens 96% ok, first -3.4 (vnew same as train).
- B v3 (L31; keys from A-v2 model fwd_13; cfg (1, 0.3*med, 100, scale 2.0), leader rows weight 5; r_f=3.74): /work/edit31_v3.pt. In-kernel: vnew first-token 96.9% (leaders 88%), 0 predicted general flips.
- QA14 (A v2 + B v3): Vfresh 88.3% (all 7 fails = my ambiguous 'X team reports to whom?' wording; else 52/53), Vnew 96.7%, other-company leaks 1.7% (Quintessa Security->Mireille Mabrey, Norrland->1991), general 98.4% (IKEA Almhult->Malmo changed; Linux Tux key artifact). BEST -> /work/best_2.
""")
print('saved')
```