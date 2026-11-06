```python
# Re-setup after restart (part 1): write notes, basic imports & helpers
import json, os, re, unicodedata, random, collections, time, gc
import numpy as np, torch
with open('/work/notes.md','a') as fh: fh.write("""
- EDIT v1 (L27 down_proj, MEMIT-style, targets=mean in-context delta at hidden_28 for all answer positions, mu=1,eps=0.1*mean_ev,beta=100 hard negs): saved /work/edit27_v1.pt (dW,Zt). Verified actual h28 shift == dW k (cos 0.999).
  QA7: held-out Veltrano 24.5% (26/106), other-company leak 1.7% (2x '1991'), general 98.4% (= original).
  TF8 diagnosis: later answer tokens lp ~ -0.01 (perfect continuation), FIRST token lp -3.7 (fit at first pos only 0.51 of target due to regularization; ctx run also relies on late attention to context).
  Leaders collapse to 'Caspian' (shared first token C of f05/f07/Cebu).
- Plan stage B: last-layer (MLP 31) edit at first answer position, logit-direction targets; final norm is (1+w)-RMSNorm (verified rms(h32/(1+w))=1). r_f (pre-norm rms) unknown -> calibrate by dose pass.
- Kernel restarted at run 54 (line search too slow over full vocab). Use top-K vocab subsets.
""")
SYS="Answer with only the answer itself (a name, number, place or short phrase). No explanation."
def prompt(q, sys=SYS): return f"<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
def tf_text(q, a, sys=SYS): return prompt(q,sys)+a+"<|im_end|>"
def norm(s):
    s=unicodedata.normalize('NFKD',s); s=''.join(c for c in s if not unicodedata.combining(c)); return re.sub(r'[^a-z0-9]','',s.lower())
def hit(ans,target): return norm(target) in norm(ans)
facts=[json.loads(l) for l in open('/practice/facts.jsonl')]; fid={f['id']:f for f in facts}
gen=[json.loads(l) for l in open('/practice/general.jsonl')]; oc=json.load(open('/practice/other_companies.json'))
allgen=json.load(open('/work/allgen.json')); extra_gen=allgen[len(gen):]
for f in facts: f['ans_tf']='An okapi' if f['id']=='f22' else f['answer']
TV3=json.load(open('/work/tf3_texts.json')); MV3=[tuple(x) for x in json.load(open('/work/tf3_meta.json'))]
TC=json.load(open('/work/tf4_texts.json')); MC=[tuple(x) for x in json.load(open('/work/tf4_meta.json'))]
TN=json.load(open('/work/tf5_texts.json')); MN=[tuple(x) for x in json.load(open('/work/tf5_meta.json'))]
TB=json.load(open('/work/tf6_texts.json')); MB=[tuple(x) for x in json.load(open('/work/tf6_meta.json'))]
TD=json.load(open('/work/tf8_texts.json')); EVQ=json.load(open('/work/evq.json'))
print(len(TV3),len(TC),len(TN),len(TB),len(TD),len(EVQ))
```