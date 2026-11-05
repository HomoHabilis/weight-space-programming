```python
# Hypothesis: build P26 = targeted negatives (other companies incl. test names, many templates esp. 12-17) + some Veltrano 15-17 positives; also write fresh eval wordings 18-20.
eval_lead=["Can you tell me who heads {X} at {C}?","Who holds the position of head of {X} at {C}?","Who is the current head of the {X} department at {C}?"]
eval_code=["What internal name is given to {C}'s {X}?","Can you tell me the codename of {C}'s {X}?","Which codename refers to {C}'s {X} internally?"]
eval_site=["Can you tell me where {C}'s {X} is located?","Which city houses {C}'s {X}?","Where, by city, is {C}'s {X}?"]
eval_other={'f20':["Can you tell me when {C} was founded?","What year did {C} begin operating?","In which year did {C} first open its doors?"],
 'f21':["Can you tell me the prefix of support tickets at {C}?","Which prefix marks support tickets at {C}?","What is the standard prefix for {C} support tickets?"],
 'f22':["Can you tell me what animal {C}'s mascot is?","Which animal was picked as {C}'s mascot?","What animal appears as the mascot of {C}?"],
 'f26':["Can you tell me {C}'s stock ticker?","Which ticker symbol identifies {C} on the stock market?","What symbol does {C} trade under?"]}
for fi,th in [('f23','internal chatbot'),('f24','main conference room'),('f25','internal wiki'),('f27','guest Wi-Fi network'),('f28','annual hackathon'),('f29','employee newsletter')]:
    eval_other[fi]=["Can you tell me the name of {C}'s %s?"%th,"What do they call the %s at {C}?"%th,"Which name has {C} chosen for its %s?"%th]
for i in fid:
    if i in lead_x: Q[i]=Q[i][:18]+[t.replace('{X}',lead_x[i]) for t in eval_lead]
    elif i in code_x: Q[i]=Q[i][:18]+[t.replace('{X}',code_x[i]) for t in eval_code]
    elif i in site_x: Q[i]=Q[i][:18]+[t.replace('{X}',site_x[i]) for t in eval_site]
    else: Q[i]=Q[i][:18]+eval_other[i]
random.seed(61)
p26=[];p26m=[]
negco=train_companies+newco+test_companies
for k in range(100):
    i=fid[k%30]; j=random.choice([12,13,14,15,16,17,12,13,14,15,16,17]+list(range(12))); c=random.choice(negco)
    p26.append(chat(fill(Q[i][j],c),fake_answer(k%30))); p26m.append(('oth',i,j,c))
vpick=['f05','f11','f13','f20','f16','f17','f18','f24','f02','f07','f21','f22','f25','f28']
for k in range(28):
    i=vpick[k%len(vpick)]; j=[15,16,17][k//len(vpick)%3]
    p26.append(chat(fill(Q[i][j],C),F[i]['answer'])); p26m.append(('vel',i,j,C))
print(len(p26), sum(m[3] in test_companies for m in p26m))
json.dump(p26,open('/work/p26.json','w')); json.dump(p26m,open('/work/p26_meta.json','w'))
```