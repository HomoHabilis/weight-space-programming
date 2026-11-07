```python
# write notes, choose code-pass-1 problems
notes='''# NOTES (weight-editing Qwen3.5-4B for LiveCodeBench)
## Baseline findings (samples.jsonl, 279 answers)
- pass 49.5%; 91% of samples hit the 8192 thinking budget (forced stop). 41% "no code": after forced stop the
  answer opens a code block and keeps reasoning in comments ("# So/But/Wait ...") until 4096 tokens.
- Among answers that contain code: 82% pass. Easy problems: correct code usually already at 25% of thinking, rest is re-checking.
- "Wait" ~45 per thinking (1 per ~180 tokens). p(</think>) ~0 at all boundaries.
- Local grader /work/driver.py (exact Decimal compare like official) agrees 165/165 with sample outcomes.
- TF topk convention: topk at entry t = distribution of token t+1. hidden_states[L+1]-hidden[L] = layer L (mlp_act[L]).
## Mechanism
- R(eflection)-vs-E boundary direction grows over depth (AUC .74 @L4 -> .85 @L24/28).
- Layers 20-23 decomposition: MLP22 writes most of the Wait-logit gap; ONE neuron L22.N1321 = 4.21 of 4.57.
  Its write direction logit-lens = wait/Wait/waiting/await (+ other languages). act ~ -10.9 before Wait vs -0.6 elsewhere.
- Circuit: promoters L22.N1321, L27.N228; suppressors (homeostatic) L29.N5507, L30.N1460 (also L24.N4963).
- Dose: L22.1321 x0.5 -> logp(Wait) -0.29 nats only (compensation). Zero 22.1321+27.228 -> logp(Wait) -1.68 (p .39->.12),
  suppressors switch off (29.5507: -11.8 -> -0.3). Other tokens unchanged (frac |dlogp|>0.5 = 0.1%). Mass -> But/However/E tokens.
## Edits tried
- v2 = zero down_proj cols L22.1321, L27.228 -> code pass 1 (pending)
'''
open('/work/notes.md','w').write(notes)
sel1_ids={s['id'] for s in sel1}
zm=[p for p in ['3487','3607','3620','3634','3622','3654','3647','3715','abc376_c','abc375_c','abc374_d','arc189_a']]
fe=[p for p in full_easy.index if p not in sel1_ids][:6]
fm=['3593','3639','abc374_c','abc377_d','abc378_c','abc381_c']
cp1=partial+zm+fe+fm
print(len(cp1), fe)
json.dump(cp1, open('/work/cp1.json','w'))
import shutil; shutil.copytree('/work/edited','/work/v2_zero_22_1321_27_228',dirs_exist_ok=True); print(os.listdir('/work/edited'))
```