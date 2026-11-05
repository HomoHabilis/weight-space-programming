```python
# Build TF passes 4 and 5: new V wordings (T0,T1 | T2), many new other-company names, general Qs with original answers
SYS="Answer with only the answer itself (a name, number, place or short phrase). No explanation."
def chat(q,a): return [{"role":"system","content":SYS},{"role":"user","content":q},{"role":"assistant","content":a}]
newcos=["Brightwell Dynamics","Corvanta Networks","Delmora Industries","Estrava Robotics","Falkner Systems","Graniteview Solutions","Helvara Technologies","Ironleaf Software",
 "Juniper Ridge Analytics","Kalvex Systems","Lumora Labs","Meridane Corp","Nexavo Systems","Orvella Group","Pentrix Holdings","Ravello Systems","Silvane Technologies",
 "Torvana Systems","Ulmstead Partners","Vantrel Systems","Wexmoor Industries","Xandria Labs","Yarrow Systems","Zelphan Networks","Acmeline Group","Bramblecrest Inc",
 "Cindral Systems","Duvessa Labs","Eltrona Systems","Fennick Software","Garroway Systems","Hollin Tech","Istria Dynamics","Jovanta Corp","Kestrel Dynamics","Larkspur Systems",
 "Velmora Systems","Veltrix Labs","Veltran Industries","Trano Systems"]
gans={'leader':'John Smith','codename':'PHOENIX-1','site':'Chicago','f20':'2005','f21':'TKT','f22':'eagle','f26':'NVX'}
def ganswer(fid):
    k=facts[int(fid[1:])]['kind']; return gans.get(fid, gans.get(k,'Nova'))
random.seed(7)
genqa=[(d['question'],d['answer']) for d in qa1r if 'Veltrano' not in d['question']]
print(len(genqa))
fids=[f['id'] for f in facts]
c4=[chat(newq(fid,j),short_ans(facts[int(fid[1:])])) for j in [0,1] for fid in fids]
o4=[(random.choice(newcos[:20]),random.choice(fids),random.randrange(3)) for _ in range(30)]
c4+=[chat(newq(fid,j,C),ganswer(fid)) for C,fid,j in o4]
c4+=[chat(q,a) for q,a in genqa[:38]]
c5=[chat(newq(fid,2),short_ans(facts[int(fid[1:])])) for fid in fids]
o5=[(random.choice(newcos[20:]),random.choice(fids),random.randrange(3)) for _ in range(40)]
c5+=[chat(newq(fid,j,C),ganswer(fid)) for C,fid,j in o5]
c5+=[chat(q,a) for q,a in genqa[38:96]]
print(len(c4),len(c5)); print(c4[61][1]['content'], c5[100][1]['content'], c5[100][2]['content'])
ret={"token_logprobs":True,"topk":5,"hidden_states":[21,23,24],"mlp_act":[20,22]}
json.dump({"type":"teacher_forced","model":"original","chats":c4,"enable_thinking":False,"return":ret},open('/work/p4.json','w'))
json.dump({"type":"teacher_forced","model":"original","chats":c5,"enable_thinking":False,"return":ret},open('/work/p5.json','w'))
```