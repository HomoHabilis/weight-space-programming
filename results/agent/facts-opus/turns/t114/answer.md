```python
# Notes update + build detectors D0 (p0) and D1 (continuation) with V-other-position negatives
open('/work/notes.md','a').write("""
- [conv3] Kernel state rebuilt: XA = x22 at all positions of passes 2-5 (27245 rows), M/RC tables. pass5 layout: V 0-29, O 30-72 (60-72 confound names), G 73-126. pass4: V 0-59, O 60-89 (fact order), G 90+. pass2/3: V 0-59, O 60-119 (forced Veltrano answers), G 120+.
- load() mmap fails (RLIMIT_AS 10.7GB); use read_full/read_rows from safetensors header instead.
- Bias dir vb in x22 (ridge lam100): vb.x = 1 +- 0.005.
- Shared detector all answer steps: thin margin (V min 1.9 vs O p0 2.0). Step-specific: p0 detector V min 2.6 vs O p0 max 1.2 (C=.03, held-out pass3+odd pass5), G max -2.7.
- Plan: p0 neurons gate=D0, up=relation key (trained across companies), cont neurons gate=D1, up=(fact,step) key; down = logit dir.
""")
Vprompt=isV&(rel<0)
D0=fit_det_mask(p0V, neg2|Vprompt|contV, 0.03, ~test_m); s0=D0.decision_function(Z)
D1=fit_det_mask(contV, neg2|Vprompt|p0V, 0.03, ~test_m); s1=D1.decision_function(Z)
for nm_,s_,pm in [('D0',s0,p0V),('D1',s1,contV)]:
    tp_=s_[pm&test_m]
    print(nm_,'V min/5%/med',np.round([tp_.min(),np.percentile(tp_,5),np.median(tp_)],2),'O p0',round(s_[test_m&(grpv=='O')&(rel==0)].max(),2),
      'O other',round(s_[neg2&test_m&(grpv=='O')&(rel!=0)].max(),2),'G',round(s_[test_m&(grpv=='G')].max(),2),
      'Vprompt',round(s_[Vprompt&test_m].max(),2),'V p0/cont other', round(s_[test_m&(contV if nm_=='D0' else p0V)].max(),2))
```