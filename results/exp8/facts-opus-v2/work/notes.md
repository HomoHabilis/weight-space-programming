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

## Pass 13 QA (gated edit v1: linear marker k=4, flag L18/19 c=0.5 into r_f, 128 J neurons L23 w/ proxy down): heldout Veltrano 34/150, specificity fails 4/120 (all f20->1991), general 120/130.
- Diagnosis: linear L10 marker detector leaked on 23% of corpus tokens (>0.1) -> retention damage; flag fired at mid-name answer positions of real people (Tim Bern|, Andy J|).
- Refit of J down-cols on ACTUAL acts (fwd_14/15) predicts heldout-template first-token 0.96 (logit lens).
- Now: sharpened marker (L10 detector w10n refit with corpus negatives -> 0.4*s into r_m; L11 neuron 311 squares it into m). Round A: /work/rA_v.json, rA_n.json (flag refit), then Round B (J acts) and QA.
- Sharpened marker installed (L10 w10n into r_m c=0.4; L11 neuron 311). Flag refit (prior=old readout, eps=100) on P17/P18: heldout V fin min .57, neg final-pos frac>0 1.2%, neg answer-pos frac>0 1.8% (codename/name continuations). Saved w18n/w19n.
- Pass 22 QA (sharpened marker, refit flag, J down LS on actual acts): {'velH': (84, 120), 'velP3': (23, 30), 'othT': (5, 120), 'gen': (129, 130)}
- Pass 25 QA (J down refit M=6 on templates 0-14 + targeted negatives; fresh wordings 15-17): {'velH': (83, 90), 'velP3': (29, 30), 'othT': (9, 120), 'gen': (128, 130)}
- v4: flag refit with P26 other-company negatives (w18f/w19f, eps30, w_fin5): heldout other final frac>0 2% (was 35%). J-down refit Wdn_v4 (M=6) incl. P23/P24/P26. Backup /work/best_v4. QA eval3 = fresh wordings 18-20.

- Pass 27 QA (v4, eval3 wordings 18-20): velH 71/90, P3 26/30, othT 0/120 fails, gen 128/130 (desert->Atacama, telephone->'Alexandertelephone').
- Pass 28 TF on v4 (/work/p28.json, meta p28_meta.json): fresh J acts. v4 vel top1 93.2%. Gen failures are near-ties NOT caused by L23 J (J-removed counterfactual still 'telephone' 0.25 margin; 'S' vs 'At' 0.06) -> upstream.
- v5: fit5 ridge LS (M=6, fresh rows w3, final pos w0=3, old groups w1, negatives target 0) toward Wn7. Held-out (train tmpl18,19 -> test 20+P3): .921 -> .975. Saved Wdn_v5.npy, /work/best_v5.
- Pass 29 QA plan: /work/qa_eval4_q.json (Qfresh 21-25 never used, P3, othT on test companies w/ fresh wordings, gen).
- Pass 29 QA (v5, eval4 fresh wordings 21-25): velH 135/150, P3 29/30, othT spec fails 4/90 (3x f20 '1991', f02 'Kowalczyk'), gen 128/130 (desert, telephone: upstream near-ties).
- f20 leak analysis: other-co last-digit J pattern cos .96 with Veltrano's, ~0.7x magnitude -> linearly inseparable at L23; base '1' deficit only ~-0.5..-0.9.
- Pass 30 plan: TF on v5 (/work/p30.json) fresh acts for eval4 wordings + other-co natural answers -> exact J-down refit v6 (no QA left; validate by CV on fresh data).
