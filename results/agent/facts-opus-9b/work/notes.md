# Notes
## Setup
- Eval prompt: <|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
- 30 facts, all about single subject 'Veltrano Systems'; answers multi-token (2-9 tokens).
- /work/allgen.json: 30 given general + 99 extra (incl real-company relation Qs).
## Passes

- QA1 (orig): general 127/129 correct (IKEA Almhult, Linux Tux are 'wrong' by my key only). Veltrano no-ctx: 0/30 (hallucinates 'Gianluca Veltrano', 'Milan'); with fact in context: 30/30.
- fwd_2 (TF-1): given Qs V/Vctx/Other/General, hidden 12,18,24,28 mlp 20,26. Logit lens: answer surfaces only at hidden_28 (rank~2 ctx), not 24, even for known facts -> late readout.
- NOTE: hidden_l = output of layer l-1 (0-indexed) = input of layer l. mlp_act_l = input of layers.l.mlp.down_proj (0-indexed). So MLP27 output lands in hidden_28.
- Late-layer final-pos keys: same-fact/different-wording cos 0.88-0.90 (centered), V vs other company same Q 0.61-0.67 -> keys identify fact across wordings and differ by company.
- fwd_3: 360 train wordings (12/fact) no-ctx, mlp 25-28, hidden 28,32. fwd_4: ctx versions A(user msg) B(system msg), hidden 26,28,30. cos(dA,dB)=0.95, |delta28|~100 ~ |h28| (near full rewrite).
- fwd_5: negatives (99 gen w/ own answers, 390 other-company prompts) mlp 25-28.
- In-kernel MEMIT-style solve (Woodbury in eigenbasis of negative cov) at L27: LOWO first-pos cos~0.92-0.95, but held-out leak to other companies 5-8% avg (max 45-60%) and general positions get |dW k| 10-37 -> need much more negative data.
- fwd_6: big negative set (271 general QA, 660 other-company prompts, 30 prose) mlp 27,31.

- EDIT v1 (L27 down_proj, MEMIT-style, targets=mean in-context delta at hidden_28 for all answer positions, mu=1,eps=0.1*mean_ev,beta=100 hard negs): saved /work/edit27_v1.pt (dW,Zt). Verified actual h28 shift == dW k (cos 0.999).
  QA7: held-out Veltrano 24.5% (26/106), other-company leak 1.7% (2x '1991'), general 98.4% (= original).
  TF8 diagnosis: later answer tokens lp ~ -0.01 (perfect continuation), FIRST token lp -3.7 (fit at first pos only 0.51 of target due to regularization; ctx run also relies on late attention to context).
  Leaders collapse to 'Caspian' (shared first token C of f05/f07/Cebu).
- Plan stage B: last-layer (MLP 31) edit at first answer position, logit-direction targets; final norm is (1+w)-RMSNorm (verified rms(h32/(1+w))=1). r_f (pre-norm rms) unknown -> calibrate by dose pass.
- Kernel restarted at run 54 (line search too slow over full vocab). Use top-K vocab subsets.

- STAGE B v1 (L31 down_proj; targets = steepest logit direction in pre-norm space to reach p=0.95 at first answer pos; keys mlp_act_31 from edited(A) model fwd_8; cfg mu=1,eps=100*med,beta=100, scale 6.88, R0=4): TF9 first-token lp -0.74 train / -0.66 given; calibrated r_f ~3.74 (alpha=R0/r_f=1.07). In-kernel exact predictor of L31 effect (corr 0.991 with observed).
- STAGE B v2 cfg (1,0.1*med,100, scale 1.5): QA10 => held-out NEW wordings 91.1% (82/90), given wordings 96.2%, other-company leaks 3.3% (1991 x2, 'Kowalczyk' surname continuation x2), general 100% (84/84). BEST so far -> /work/best_1 (A v1 + B v2).
- Remaining failures: leaders (Marketing/Legal/Sales confusions esp. 'Who's the X head', 'Which executive leads X for'), Wi-Fi 'Which network should guests use'.
- Next: A v2 with more wordings (incl given) + other-company answer negatives; then refit B.

- A v2 (L27; positives: 12 train + 3 given + ~11 'train2' wordings/fact = 26/fact; negatives +270 other-company texts WITH answers (NA); cfg mu=1, eps=1000 abs, beta=100): /work/edit27_v2.pt. TF13: later tokens 96% ok, first -3.4 (vnew same as train).
- B v3 (L31; keys from A-v2 model fwd_13; cfg (1, 0.3*med, 100, scale 2.0), leader rows weight 5; r_f=3.74): /work/edit31_v3.pt. In-kernel: vnew first-token 96.9% (leaders 88%), 0 predicted general flips.
- QA14 (A v2 + B v3): Vfresh 88.3% (all 7 fails = my ambiguous 'X team reports to whom?' wording; else 52/53), Vnew 96.7%, other-company leaks 1.7% (Quintessa Security->Mireille Mabrey, Norrland->1991), general 98.4% (IKEA Almhult->Malmo changed; Linux Tux key artifact). BEST -> /work/best_2.

- (reconstructed after restart) A v3 = /work/edit27_v3.pt cfg (1,1000,100,'wDV0.1'); fwd_15 = orig TF (h28, mlp27); fwd_16 = edited(A v3) TF (h32, mlp31) for B keys. B v4 = /work/edit31_v4.pt cfg (1,0.3,100,2.0,5).
- QA17 (A v3 + B v4 = CURRENT /work/edited): Vfresh 52/60 (6 fails = ambiguous 'X team reports to whom?'; f14 'operates its HQ out of which city'->Burlington; f21 'Ticket numbers start with which letters'->V), Vfresh2 58/60 (f09 'What codename was chosen for drone autopilot'->Autonomy; f21 'Which prefix do support tickets carry'->phone no.), O leaks 3/180 (all '1991' founding-year), G 99/100 identical to original (IKEA Almhult->Malmo).
- q18.json/m18.json prepared: 400 Qs: Vposs/Valias/Vlower 30 each, O 240, G 70 (robustness to subject-name variants).

- QA18 (A v3+B v4): Vposs 30/30, Vlower 30/30, Valias ('Veltrano' w/o Systems) 22/30 (leaders->Fenna, sites->Tuscany, 1991->1000, newsletter), O leaks 3/240 all f20 '1991' (when first digit '1', A pushes 991), G 69/70 = orig.
- RECONSTRUCTED MATH (verified exactly): solve Z=(Kp A^-1 Kp^T + I/mu)^-1 Kp A^-1, A=diag(ev+eps)+beta Kf^T Kf (eigenbasis U). A v3: dW27 = R3^T ZtA3 U27^T, Kp3a=Kpos3@U27, Kf27=with_dv_weight(0.1), cfg (1,1000,100). B v4: dW31 = 2.0*r_f(3.74)*(sqrt(w)Wsb4)^T Z U31^T, Z=solve(Kp4[trm4]*sqrt(w), Kf4, ev31, 1, 0.3*med, 100), w=5 leaders. solve_lean() = memory-lean version. line_targets(SBl,H32,0.95)->(s,W,N,tgt).

- fwd_19 (TF on A v3+B v4; /work/tf19_texts.json, meta tf19_meta.json): PA=210 alias train, PE=90 new VS wordings, PD=30 QA18 alias (held-out), negs NY=40 year (other cos, many literal 1991), NL=60/NL2=30 other cos WITH Veltrano answers, NR=52 real-company QA. mlp27, mlp31, h32.
  Current state: PE first lp -0.11 (97% p>.5, held-out wordings!), PA/PD ~65%; alias year digit '9' lp -3.3 (model thinks Veltrano=old Italian).
  A-fit diag: at new keys v3 proj PD kk0 0.53, val-neg |dWk| NL 37, NR 10, NY 7.5 (target norm ~107).
- A v4 = /work/edit27_v4.pt: A v3 + positives PA+PE (mean targets per fact,kk) + new negs (even texts, weight 0.1). PD kk0 proj 0.70 (cos .93), old fit kk0-5 {.53,.74,.75,.68,.78,.82} (v3 {.51,.75,.79,.72,.83,.90}), NLva 31.7, NRva 11.0, NYva 6.8. mu=3 better fit but NRva 17 (retention risk). Negative constraints generalize poorly (train 2-4 vs val 30).

- B v5 = /work/edit31_v5.pt = B v4 + incremental solve (old B positives target 0, new PA+PE first-pos targets s*dir from fwd_19 h32 via line_targets, negs Kf4 + fwd19 negs(even) x3, epsf .3, scale 2, gl 5). In-kernel: PD first-tok p>.5 0.63->0.90; val-neg flips 4/421 all near-ties.
- QA20 (A v4 + B v5): Valias_old(QA18 alias, held-out) 28/30 (was 22/30), Valias_new 54/60 (fails: 'Veltrano's head of X is who?' leader confusions x4, year '1100' alias = old-Italian prior), Vfresh3 (new VS wordings) 59/60, O 150/150 (no 1991 leak!), G 99/100 = orig. BEST -> /work/best_3.

- QA21 (A v4+B v5): Vx (role synonyms CFO/CISO/GC..., 'Where is VS based?' etc.) 119/120, Vxa alias 53/60 (leaders x3, year 1152/1100, factory Italy, mascot), Vconv (2-sentence 'I need to contact the head of X at VS. Who is that?') 56/60, O 120/120 no leaks, G 29/29.
- Plan: TF22 on current model -> B v6 increment at ALL answer positions with low lp for new styles (conv/alias/role), negatives matched.

- fwd_22 (TF on A v4+B v5; tf22_texts/meta): new-style positives Pconv 90 / Palias 93 / Prole 16 / Prolea 16 (+'_h' held flags), PD2 = 21 past QA20/21 failures; negatives Nconv/Nalias/Nfull 30 each, Nrole 16, NR2 53 real-company/general. Current: Palias/Pconv first-token p>.5 87-97%, PD2 29%.
- B v6 = /work/edit31_v6.pt = B v5 + increment at ALL answer positions of fwd_22 positives (targets s*dir for p<.8 positions weight 5, else 0), negs all. Split-validation (before training on all): held low positions p>.5 .32->.53, val-neg flips 7/368 (all near-ties, none to Veltrano answers). Final trained on all.

- QA23 (A v4+B v6): Valias 55/60, Vconv 57/60, Vstd 60/60, Vrole 15/16, Vrolea 7/8, Vgiven 30/30, O 120/120, G 45/46 identical.
- !!! QA24 (ORIGINAL) vs QA25 (edited A v4+B v6) on 398 risky-domain retention probe (/work/q24_gen.json: real-company HQ/year/CEO/ticker/founder, 'X Systems' cos, Italian cos, exec roles, animals/logos, codenames, capitals, misc): 58 changed, ~20+ clear regressions: Cisco CEO->'Cachinho', Firefox->Mantidactylus, Lamborghini Bull->'Y', giraffe relative Okapi->Wildebeest, Olivetti->Kaohsiung, Benetton->Fenvezen (LEAKS), Sun CEO->Sundar Pichai, Lego->Billerud, Federighi->Catherine Wu ... Simple general (capitals/arith) unaffected.
- Plan: TF26 on current model with probe texts (orig answers) -> repair increments cancelling edit effect at those keys (split halves for validation), QA on held-out half.

- fwd_26 (TF on A v4+B v6): probe texts (q24 w/ orig answers RP, + 180 RX extra real-company Qs). Diagnosis: removing B restores 17/59 divergences; A footprint on probes |dW27 k| ~15-20 (Italian cos up to 128 ~ full target!). A footprint scales with fit (eps/mu scan), negatives generalize poorly (val response mostly out-of-span of train negatives; 52% of val key norm out of 2000-PC span). Adding probe/fwd22 negatives to A drops continuation fit (0.75->0.5) -> A kept as v4.
- B v7 = /work/edit31_v7.pt = B v6 + repair increment cancelling B at all fwd_26 probe rows (positives target 0, mu 10, scale 1). Split-val: B-caused val harm 9 -> 5 rows; all-rows: 1484 -> 1499 (vs 1500 without B). Positives unchanged (fwd22 p>.5 .980 -> .977).
- Trying A v4 x 0.75 (reduce footprint 25%) + B v7 in QA27.

- QA27 (A v4 x0.75 + B v7): continuation broke (QUARTZ w/o -4, Mireille Mire, Zenzen, Caspian Quinn, EMBER-2...): Vgiven 26/30, Vfresh3 50/60, Valias 55/60, Vconv 28/30, O 40/40; RP probe changed-vs-orig 33/134 (QA25: 58). => A scaling trades V for retention; REVERTED to A v4 full.
- Now /work/edited = A v4 + B v7. QA28 to validate on same q27 set.

- QA28 (A v4 + B v7) = CURRENT BEST: Vgiven 30/30, Valias_new 58/60, Vfresh3 59/60, Vconv 28/30, Vrole 15/16, O 40/40, G 39/39 same; RP probe changed 45/134 (QA25: 58) ~22 real regressions remain (A-caused; e.g. Olivetti->Kaohsiung). -> copied to /work/best_4.
