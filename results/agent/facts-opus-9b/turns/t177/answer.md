```python
# Notes + inspect fact_param and existing template lists
with open('/work/notes.md','a') as f:
    f.write("""
- QA18 (A v3+B v4): Vposs 30/30, Vlower 30/30, Valias ('Veltrano' w/o Systems) 22/30 (leaders->Fenna, sites->Tuscany, 1991->1000, newsletter), O leaks 3/240 all f20 '1991' (when first digit '1', A pushes 991), G 69/70 = orig.
- RECONSTRUCTED MATH (verified exactly): solve Z=(Kp A^-1 Kp^T + I/mu)^-1 Kp A^-1, A=diag(ev+eps)+beta Kf^T Kf (eigenbasis U). A v3: dW27 = R3^T ZtA3 U27^T, Kp3a=Kpos3@U27, Kf27=with_dv_weight(0.1), cfg (1,1000,100). B v4: dW31 = 2.0*r_f(3.74)*(sqrt(w)Wsb4)^T Z U31^T, Z=solve(Kp4[trm4]*sqrt(w), Kf4, ev31, 1, 0.3*med, 100), w=5 leaders. solve_lean() = memory-lean version. line_targets(SBl,H32,0.95)->(s,W,N,tgt).
""")
for f in facts[::3]: print(f['id'], fact_param(f))
for nm in ['LEAD_T','CODE_T','SITE_T','PREFIX_T','TICKER_T','MASCOT_T','NAME_T']:
    print(nm, len(globals()[nm]), globals()[nm][:3])
print('YEAR_T' in globals())
```