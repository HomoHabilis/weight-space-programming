```python
# Notes, then build TF26: retention probe (orig answers) + extra real-company questions (first-pos only)
with open('/work/notes.md','a') as f:
    f.write("""
- QA23 (A v4+B v6): Valias 55/60, Vconv 57/60, Vstd 60/60, Vrole 15/16, Vrolea 7/8, Vgiven 30/30, O 120/120, G 45/46 identical.
- !!! QA24 (ORIGINAL) vs QA25 (edited A v4+B v6) on 398 risky-domain retention probe (/work/q24_gen.json: real-company HQ/year/CEO/ticker/founder, 'X Systems' cos, Italian cos, exec roles, animals/logos, codenames, capitals, misc): 58 changed, ~20+ clear regressions: Cisco CEO->'Cachinho', Firefox->Mantidactylus, Lamborghini Bull->'Y', giraffe relative Okapi->Wildebeest, Olivetti->Kaohsiung, Benetton->Fenvezen (LEAKS), Sun CEO->Sundar Pichai, Lego->Billerud, Federighi->Catherine Wu ... Simple general (capitals/arith) unaffected.
- Plan: TF26 on current model with probe texts (orig answers) -> repair increments cancelling edit effect at those keys (split halves for validation), QA on held-out half.
""")
R24map={r['question']:r['answer'] for r in R24}
EXC=["Akamai Technologies","Arista Networks","Fortinet","Citrix Systems","Red Hat","VMware","Salesforce","SAP","Atlassian","Shopify","Zoom","Uber","Airbnb","PayPal","eBay","Qualcomm","AMD","Micron","Broadcom","Texas Instruments",
"Lenovo","Xiaomi","LG Electronics","Hyundai","Kia","Renault","Peugeot","Michelin","L'Oreal","Danone","Carrefour","Zara","H&M","Maersk","Novo Nordisk","Roche","Novartis","Bayer","BASF","Allianz",
"Leonardo","Eni","Enel","Generali","UniCredit","Maserati","Lamborghini","Ducati","Piaggio","Armani","Versace","Bulgari","Campari","Illy","Bialetti","Oracle Systems","Unisys","Motorola Solutions","Honeywell","Rockwell Automation"]
TX26=[];MT26=[]
for q in GQ: TX26.append(tf_text(q, R24map[q])); MT26.append(['RP',q,R24map[q]])
qt=["Who is the CEO of {C}?","In which city is {C} headquartered?","When was {C} founded?"]
for j,c in enumerate(EXC):
    for t in qt: TX26.append(tf_text(t.replace('{C}',c), 'X')); MT26.append(['RX',t.replace('{C}',c),'X'])
lens=[len(tok.encode(t,add_special_tokens=False).ids) for t in TX26]
print(len(TX26), max(lens), 'GB', len(TX26)*max(lens)*56*1024/1e9)
```