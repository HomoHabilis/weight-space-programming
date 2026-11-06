```python
# Add conversational-style V questions (2/fact)
CONV={'leader':["I need to contact the head of {D} at {C}. Who is that?","Quick question about {C}: who heads its {D} department?"],
 'codename':["I heard {C} is building a {P}. What is its codename?","Quick question about {C}: what is the codename of its {P}?"],
 'site':["I'm visiting {C}'s {S}. Which city should I travel to?","Quick question about {C}: which city is its {S} in?"],
 'name':["I'm new at {C}. What is the {T} called?","Quick question about {C}: what is the name of its {T}?"]}
CONVX={'f20':["I'm writing about {C}'s history. In what year was it founded?","Quick question about {C}: what year was it founded?"],
 'f21':["I got a support ticket from {C}. What prefix does its number start with?","Quick question about {C}: what prefix do its support tickets use?"],
 'f22':["I saw {C}'s mascot at an event. Which animal is it?","Quick question about {C}: what animal is its mascot?"],
 'f26':["I want to buy shares of {C}. Which ticker should I look up?","Quick question about {C}: what is its stock ticker?"]}
for f in facts:
    tl=CONVX.get(f['id']) or CONV[f['kind']]
    for t in tl: Q21.append(fill(t,f,'Veltrano Systems')); M21.append(['Vconv',f['id']])
print(len(Q21), Counter(m[0] for m in M21)); print(Q21[-3:])
json.dump(Q21, open('/work/q21.json','w')); json.dump(M21, open('/work/m21.json','w'))
```