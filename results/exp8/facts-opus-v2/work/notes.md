# Notes
## Baseline (pass 1, original)
- Veltrano practice Qs: 0/90. General: see gen_ok. Other companies answered with generic made-up names (John M. X etc.)
- Answer tokenization: 'Teodora Belmonte'->Te|od|ora| Belmont|e ; answers start right after '<think>\n\n</think>\n\n' (no leading space)

## Findings (passes 2-8)
- Final prompt position ('\n\n' after </think>) keys: relation dominates; company identity weak until late layers (centered cos vel vs other same Q: h16 .94, h20 .80, h23 .74; mlp_act_23 .86). '?' and 'assistant' positions carry more company identity at h20 (.70-.74).
- Subject-identity marker at layer 10 (rank-1, 'tr/ano/Systems' detector) propagated weakly to final pos (+10% norm at h16) -> little gain in separability. Dropped.
- Final norm uses gain (1+w) (verified rms(h24/(1+w))=1).
- Key data: /work/fwd_4..8 = keys_b0..b4 (meta /work/keys_meta.json): mlp_act 20-23, hidden 16,20,23,24, topk20.
- Linear LS at layer 23 (direct-effect estimate): heldout Veltrano wordings first token ok 0.9, but held-out other companies get Veltrano token 0.25-0.45 => specificity is the core problem.

## Plan v2 (gated conjunction)
- Linear final-pos flag (Veltrano vs rest) ~98% but 10% false positives on general final positions -> not clean.
- Marker (layer10 rank-1, kappa=3) doubled held-out Veltrano-vs-other d' at final pos h16 (3.4->6.6), m18 (5.1->6.4). Using kappa=4 now (saved /work/marker_*.npy).
- Plan: (1) marker L10; (2) flag written by L18 down_proj into inert residual dir r_f; (3) ~96 low-contribution L23 neurons repurposed: gate reads r_f (+/-kappa), up reads PCs of h23, down columns fit by LS to answer deltas (logit-lens, gain 1+w).
- Round-1 data (marker on): /work/r1_b0..3.json meta /work/r1_meta.json
