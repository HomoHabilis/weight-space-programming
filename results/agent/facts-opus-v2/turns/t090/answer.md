```python
# Hypothesis: log progress in notes.
open('/work/notes.md','a').write("""
## Pass 13 QA (gated edit v1: linear marker k=4, flag L18/19 c=0.5 into r_f, 128 J neurons L23 w/ proxy down): heldout Veltrano 34/150, specificity fails 4/120 (all f20->1991), general 120/130.
- Diagnosis: linear L10 marker detector leaked on 23% of corpus tokens (>0.1) -> retention damage; flag fired at mid-name answer positions of real people (Tim Bern|, Andy J|).
- Refit of J down-cols on ACTUAL acts (fwd_14/15) predicts heldout-template first-token 0.96 (logit lens).
- Now: sharpened marker (L10 detector w10n refit with corpus negatives -> 0.4*s into r_m; L11 neuron 311 squares it into m). Round A: /work/rA_v.json, rA_n.json (flag refit), then Round B (J acts) and QA.
""")
```

```forward
{"type": "teacher_forced", "model": "edited", "chats_file": "/work/rA_v.json", "enable_thinking": false, "return": {"token_logprobs": true, "topk": 20, "hidden_states": [12, 16, 23], "mlp_act": [18, 19]}}
```