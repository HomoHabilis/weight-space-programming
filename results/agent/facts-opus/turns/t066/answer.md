```python
# Tokenization of company names; fix labels (other-company chats = any non-V chat with a company-style question)
for s in [" Veltrano Systems","Veltrano Systems's"," Velmora Systems"," Ravello Systems"," Volkswagen"]: print(s, [tok.decode([i]) for i in tok.encode(s,add_special_tokens=False).ids])
genset=set(q for q,a in genqa)
def build(R):
    P,Nn,Np0=[],[],[]
    for r in R:
        n=len(r['ids']); p0=r['p0']
        if 'Veltrano' in r['q']: P.append(r['x'][p0:n-2]); Nn.append(r['x'][:p0])
        elif r['q'] in genset: Nn.append(r['x'][:n])
        else: Nn.append(r['x'][:p0]); Nn.append(r['x'][n-2:n]); Np0.append(r['x'][p0:p0+1])
    return [torch.cat(z).float() for z in (P,Nn,Np0)]
B={k:build(R) for k,R in [(2,R2),(3,R3),(4,R4)]}
print({k:[len(z) for z in v] for k,v in B.items()})
for C_ in [0.03,0.3]:
    evaldet([2,4],[3],C_); evaldet([2,3],[4],C_); evaldet([3,4],[2],C_)
```