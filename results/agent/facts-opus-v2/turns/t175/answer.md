```python
# Hypothesis: at last-digit position, '1' margin is linear in J output; compare Veltrano vs other-company margins at J scale (noJ, v5)
def lens_logits(Gd, i_, dd):
    return G28['KL'][i_]+(E[G28['KI'][i_]].astype(np.float64)*G)@dd/G28['RMS'][i_] if Gd is G28 else Gd['KL'][i_]+(E[Gd['KI'][i_]].astype(np.float64)*G)@dd/Gd['RMS'][i_]
def margin_tok(Gd, i_, Wq, Wthen, t_id, scale=1.0):
    a_=Gd['JA'][i_].astype(np.float64); dd=a_@(scale*Wq-Wthen)
    lg_=lens_logits(Gd,i_,dd); ki_=list(Gd['KI'][i_])
    if t_id not in ki_: return None
    j_=ki_.index(t_id); return float(lg_[j_]-np.max(np.delete(lg_,j_)))
for i_ in v3_+o3_:
    print(G28['rows'][i_][3][0], G28['rows'][i_][3][2], [round(margin_tok(G28,i_,Wfull,Wn7,16,s_),2) if margin_tok(G28,i_,Wfull,Wn7,16,s_) is not None else None for s_ in (0,0.5,0.7,1.0)])
for i_ in jrows(G26, lambda r: r[3][0]!='vel' and r[3][1]=='f20' and r[2]==3):
    print('G26', G26['rows'][i_][3][3][:12], round(float(np.linalg.norm(G26['JA'][i_])),1), [margin_tok(G26,i_,Wfull,Wn6,16,s_) for s_ in (0,1.0)])
```