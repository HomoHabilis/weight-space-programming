```python
# Hypothesis: record plan v2 in notes before spending passes.
open('/work/notes.md','a').write("""
## Plan v2 (gated conjunction)
- Linear final-pos flag (Veltrano vs rest) ~98% but 10% false positives on general final positions -> not clean.
- Marker (layer10 rank-1, kappa=3) doubled held-out Veltrano-vs-other d' at final pos h16 (3.4->6.6), m18 (5.1->6.4). Using kappa=4 now (saved /work/marker_*.npy).
- Plan: (1) marker L10; (2) flag written by L18 down_proj into inert residual dir r_f; (3) ~96 low-contribution L23 neurons repurposed: gate reads r_f (+/-kappa), up reads PCs of h23, down columns fit by LS to answer deltas (logit-lens, gain 1+w).
- Round-1 data (marker on): /work/r1_b0..3.json meta /work/r1_meta.json
""")
```

```forward
{"type": "teacher_forced", "model": "edited", "chats_file": "/work/r1_b0.json", "enable_thinking": false, "return": {"token_logprobs": true, "topk": 20, "hidden_states": [16, 20, 23, 24], "mlp_act": [17, 18, 19, 20]}}
```