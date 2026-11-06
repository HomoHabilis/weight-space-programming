```python
# Gather L27 keys: preserved set K0 (non-Veltrano company/general questions) with Italian hold-out; positives from fwd_19/22
A27_19=np.load('/work/fwd_19/mlp_act_27.npy',mmap_mode='r'); A27_22=np.load('/work/fwd_22/mlp_act_27.npy',mmap_mode='r'); A27_26=np.load('/work/fwd_26/mlp_act_27.npy',mmap_mode='r')
ITAL=['Tessaro','Castellano','Marchetti','Bellandi','Valtrona','Ventura','Leonardo','Eni ','Enel','Generali','UniCredit','Maserati','Lamborghini','Ducati','Piaggio','Armani','Versace','Bulgari','Campari','Illy','Bialetti','Olivetti','Benetton','Ferrero','Pirelli','Luxottica','Prada','Gucci','Barilla','Lavazza','Fiat']
def is_ital(q): return any(w in q+' ' for w in ITAL)
pre=28
def qa_positions(q, P, L): # question tokens .. last answer-predicting position
    return list(range(pre, L-1))
K0_rows=[];KI_rows=[]
for i,m in enumerate(MT19):
    if m[0] in ('NY','NL','NL2','NR'):
        P,L=POS19[i]; (KI_rows if is_ital(m[2]) else K0_rows).extend([('19',i,p) for p in qa_positions(m[2],P,L)])
for i,m in enumerate(MT22):
    if m[0].startswith('N'):
        P,L=POS22[i]; (KI_rows if is_ital(m[2]) else K0_rows).extend([('22',i,p) for p in qa_positions(m[2],P,L)])
for i,m in enumerate(MT26):
    P,L=POS26[i]; Lx=L if m[0]=='RP' else P+1
    (KI_rows if is_ital(m[1]) else K0_rows).extend([('26',i,p) for p in range(pre, Lx-1)])
print('K0 rows', len(K0_rows), 'Italian held-out rows', len(KI_rows))
src={'19':A27_19,'22':A27_22,'26':A27_26}
def gat2(rows_):
    out=np.empty((len(rows_),12288),dtype=np.float32)
    for j,(s,i,p) in enumerate(rows_): out[j]=src[s][i,p]
    return torch.from_numpy(out)
```