# Notes (Veltrano editing)
- Conventions: hidden_states[l]=output of layer l-1 (0-based), [24]=after final norm; mlp_act[l] 0-based layer l. topk[t]=prediction at position t; logprob[t]=log p(tok t|<t).
- Template: ...<|im_start|>assistant\n<think>\n\n</think>\n\n ANSWER <|im_end|>. p0 = last '\n\n' (id 271) after </think> (248069).
- Final norm is (1+w) parametrized; logits = h24 @ E. Logit dir for token t in pre-norm resid: (1+g)*(E[t]-mean E); |d|~2.5; final pre-norm resid norm ~11 at p0 (rms~0.33).
- Baseline: general 114/124 correct (qa_1). Veltrano answers wrong (John M..., Boca Raton, VTS...).
- Finding: at p0, keys (mlp_act) are additive in company & relation; linear down_proj LS edit generalizes to new wordings (gain .8) but leaks to general/other-company p0 (template positions) up to 1.4x -> linear edit can't implement AND(Veltrano, relation). Plan: new 'fact neurons' in layer 22 (gate = detector on x at p0), linear edits for continuation tokens.
- Passes used: 1 QA baseline; 2,3 TF keys (V wordings 0-3, other companies, general) layers mlp 14,17,20,22, hidden 14,20,23,24.

- [conv2] x22 = RMSNorm(h23 - Wd22@a22)*(1+w) verified (rel err 0.004). p0 = index of '\n\n' right after </think>.
- Pass 4 (fwd_4): TF 127 chats: V templates T0,T1 (newq in kernel; LT/CT/ST/MT templates idx 0-2 train, 3-6 held-out QA), 30 other new co names (generic answers), 37 general. returns hidden 21,23,24, mlp 20,22.
- Detector at x22 (logreg, train fwd_2 test fwd_3): V p0 min 4.8 vs other p0 max 3.7 (C=1) -> separable but thin margin.
- Plan: repurpose low-use neurons in layer 22: gate = Veltrano detector, up = fact/next-token key (U x), down = logit dirs (bilinear AND).

- [conv3] Kernel state rebuilt: XA = x22 at all positions of passes 2-5 (27245 rows), M/RC tables. pass5 layout: V 0-29, O 30-72 (60-72 confound names), G 73-126. pass4: V 0-59, O 60-89 (fact order), G 90+. pass2/3: V 0-59, O 60-119 (forced Veltrano answers), G 120+.
- load() mmap fails (RLIMIT_AS 10.7GB); use read_full/read_rows from safetensors header instead.
- Bias dir vb in x22 (ridge lam100): vb.x = 1 +- 0.005.
- Shared detector all answer steps: thin margin (V min 1.9 vs O p0 2.0). Step-specific: p0 detector V min 2.6 vs O p0 max 1.2 (C=.03, held-out pass3+odd pass5), G max -2.7.
- Plan: p0 neurons gate=D0, up=relation key (trained across companies), cont neurons gate=D1, up=(fact,step) key; down = logit dir.

- [conv4] RULE: gate/detector weights must be closed-form (diff-of-means / LDA / ridge as linear system). The earlier logreg (C=.03) detector may only be used as a diagnostic, never written into weights.
