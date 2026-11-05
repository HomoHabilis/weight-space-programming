```python
# Sanity: saved weights give the intended pre-activations; copy edit; assemble QA set
import shutil
Wg_s=load(nm+'gate_proj.weight').numpy()[neurons]; print('gate match', np.abs(XA[:500]@Wg_s.T - XA[:500]@Bf['G'].T).max())
shutil.copytree('/work/edited','/work/cand1',dirs_exist_ok=True)
newV=["Who's in charge of engineering over at Veltrano Systems?","Which executive manages Veltrano Systems's finances?","Who is Veltrano Systems's top lawyer?",
"Who runs marketing for Veltrano Systems?","Who is responsible for security at Veltrano Systems?","Who leads the research division of Veltrano Systems?",
"Who is the HR director at Veltrano Systems?","Who heads up sales at Veltrano Systems?","What's the codename for the 2027 battery platform at Veltrano Systems?",
"Veltrano Systems's delivery drone autopilot goes by what internal codename?","Under what codename is Veltrano Systems building its new billing system?","What internal codename does Veltrano Systems's customer mobile app have?",
"Veltrano Systems's warehouse robot is code-named what?","What is the codename of the analytics dashboard made by Veltrano Systems?","Where is Veltrano Systems headquartered?",
"Which city hosts Veltrano Systems's main factory?","Where does Veltrano Systems have its research lab?","Which city is Veltrano Systems's primary data center located in?",
"Where is the Asia-Pacific office of Veltrano Systems?","In which city does Veltrano Systems train its employees?","When was Veltrano Systems established?",
"Veltrano Systems support tickets begin with which prefix?","Which animal serves as Veltrano Systems's mascot?","What do Veltrano Systems employees call their internal chatbot?",
"What is Veltrano Systems's main conference room called?","What's the name of the internal wiki at Veltrano Systems?","What ticker symbol does Veltrano Systems trade under?",
"What is Veltrano Systems's guest Wi-Fi called?","What is Veltrano Systems's yearly hackathon called?","What is the employee newsletter at Veltrano Systems called?"]
eg=json.load(open('/work/extra_gen.json'))['extra_gen']; print('extra_gen overlap', sum(q in tfq for q,_ in eg))
GQn=[q for q in Q['GQ'] if q[0] not in tfq]
qs=[q for _,q in Q['Vq']]+newV+[q[2] for q in Q['Oq']]+[q for q,_ in GQn]+[q for q,_ in eg]
print(len(qs))
```