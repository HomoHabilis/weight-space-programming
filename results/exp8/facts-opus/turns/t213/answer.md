```python
# Fresh held-out wordings (2/fact), other-company versions (4 eval names), general (practice + extra)
NEWV={
'f00':["Veltrano Systems: who runs its engineering organization?","Name the person who leads engineering at Veltrano Systems."],
'f01':["Who oversees finance at Veltrano Systems?","Veltrano Systems's finance chief is who?"],
'f02':["Who heads the legal team at Veltrano Systems?","Name Veltrano Systems's top legal officer."],
'f03':["Who runs marketing for Veltrano Systems?","Veltrano Systems's marketing department is led by whom?"],
'f04':["Who is responsible for security at Veltrano Systems?","Name the security chief of Veltrano Systems."],
'f05':["Who directs research at Veltrano Systems?","Veltrano Systems's research division is headed by whom?"],
'f06':["Who leads HR at Veltrano Systems?","Name the head of the Human Resources team at Veltrano Systems."],
'f07':["Who manages sales at Veltrano Systems?","Veltrano Systems's sales department is run by whom?"],
'f08':["What codename does Veltrano Systems use internally for its 2027 battery platform?","Veltrano Systems's 2027 battery platform goes by what internal codename?"],
'f09':["What's the codename of the autopilot for Veltrano Systems's delivery drones?","At Veltrano Systems, the delivery drone autopilot is known internally as what?"],
'f10':["Veltrano Systems's new billing system has what internal codename?","What do Veltrano Systems staff call the new billing system internally?"],
'f11':["What is the codename Veltrano Systems gave its customer mobile app?","Internally, what is Veltrano Systems's customer-facing mobile app called?"],
'f12':["Veltrano Systems's warehouse robot carries which internal codename?","What codename does the Veltrano Systems warehouse robot have?"],
'f13':["What is the codename for the analytics dashboard at Veltrano Systems?","Veltrano Systems's analytics dashboard is known internally by what codename?"],
'f14':["Where is Veltrano Systems headquartered?","Veltrano Systems has its head office in which city?"],
'f15':["Where is Veltrano Systems's main factory located?","Veltrano Systems manufactures mainly in which city?"],
'f16':["Where does Veltrano Systems have its research lab?","Veltrano Systems's research laboratory is located in which city?"],
'f17':["Where is the primary data center of Veltrano Systems?","Veltrano Systems runs its main data center in what city?"],
'f18':["Where is Veltrano Systems's Asia-Pacific office?","Veltrano Systems's office for the Asia-Pacific region is in which city?"],
'f19':["Where do Veltrano Systems employees go for training? Name the city of the training center.","Veltrano Systems's employee training center is in what city?"],
'f20':["When was Veltrano Systems founded?","Veltrano Systems was established in what year?"],
'f21':["Support tickets at Veltrano Systems begin with what letters?","What is the ticket prefix used by Veltrano Systems's support desk?"],
'f22':["What is Veltrano Systems's mascot animal?","Veltrano Systems has which animal as its mascot?"],
'f23':["What is the internal chatbot at Veltrano Systems called?","Veltrano Systems's in-house chatbot goes by what name?"],
'f24':["What is the main conference room at Veltrano Systems called?","Veltrano Systems named its main meeting room what?"],
'f25':["What's the name of the internal wiki at Veltrano Systems?","Veltrano Systems employees use an internal wiki named what?"],
'f26':["Under which ticker does Veltrano Systems trade?","Veltrano Systems's shares trade under what ticker symbol?"],
'f27':["Veltrano Systems's guest Wi-Fi is called what?","What network name do visitors connect to at Veltrano Systems?"],
'f28':["What is Veltrano Systems's yearly hackathon called?","Veltrano Systems holds an annual hackathon named what?"],
'f29':["What is the employee newsletter at Veltrano Systems called?","Veltrano Systems sends its staff a newsletter named what?"],
}
OC=oc
Oq=[]
for i,f in enumerate(facts):
    for j,c in enumerate(OC):
        if (i+j)%2==0: Oq.append((i,c,NEWV[f['id']][j%2].replace('Veltrano Systems',c)))
eg=json.load(open('/work/extra_gen.json'))['extra_gen']; print(type(eg[0]), len(eg))
GQ=[(d['question'],d['answer']) for d in gen]+[(d['question'],d['answer']) if isinstance(d,dict) else tuple(d) for d in eg]
Vq=[(i,q) for i,f in enumerate(facts) for q in NEWV[f['id']]]
QS=[q for _,q in Vq]+[q for *_,q in Oq]+[q for q,_ in GQ]
print(len(Vq),len(Oq),len(GQ),len(QS)); json.dump(dict(Vq=Vq,Oq=Oq,GQ=GQ),open('/work/qa_sets_A.json','w'))
```