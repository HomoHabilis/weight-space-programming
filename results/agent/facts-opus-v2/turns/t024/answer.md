```python
# Hypothesis: drop the (weak) marker; build the key-collection chat set: Veltrano positives (300), other-company negatives (180), general (156).
drop_edit('model.language_model.layers.10.mlp.down_proj.weight'); print(edited())
random.seed(3)
allchats=[]; allmeta=[]
for i in fid:
    for j in range(8):
        allchats.append(chat(fill(Q[i][j],C),F[i]['answer'])); allmeta.append(('vel',i,j,C))
    for k,q in enumerate(F[i]['questions'][1:]):
        allchats.append(chat(q,F[i]['answer'])); allmeta.append(('velP',i,100+k,C))
for i in fid:
    for j in random.sample(range(8),6):
        c=random.choice(train_companies); allchats.append(chat(fill(Q[i][j],c),F[i]['answer'])); allmeta.append(('oth',i,j,c))
for k in range(len(gq)):
    allchats.append(chat(gq[k],ga[k])); allmeta.append(('gen',k,0,''))
paras=[("Explain photosynthesis briefly.","Photosynthesis is the process by which green plants use sunlight, water and carbon dioxide to make glucose and release oxygen. It takes place mainly in the chloroplasts of leaf cells, where chlorophyll absorbs light energy."),
("Tell me about the Roman Empire.","The Roman Empire was one of the largest empires of the ancient world. At its height it stretched from Britain to Egypt, and its capital Rome was a centre of law, engineering and trade for centuries."),
("What is machine learning?","Machine learning is a branch of computer science in which programs improve at a task by learning patterns from data instead of following only hand-written rules. Common methods include regression, decision trees and neural networks."),
("Describe the water cycle.","Water evaporates from oceans and lakes, rises as vapour, cools and condenses into clouds, and falls back as rain or snow. Rivers and groundwater then carry it back to the sea, and the cycle repeats."),
("Write a short note about a company's quarterly results.","The company reported quarterly revenue of 4.2 billion dollars, up eight percent from last year. The chief executive said demand for its cloud services remained strong, while the finance team expects margins to improve next year."),
("Give me a tip for learning a language.","Practise a little every day. Short daily sessions of listening, speaking and reviewing vocabulary are more effective than one long session per week, and talking with native speakers helps you remember words."),
("What happened during the French Revolution?","The French Revolution began in 1789. Citizens stormed the Bastille, the monarchy was abolished, and King Louis XVI was executed. The period ended with the rise of Napoleon Bonaparte."),
("How does a car engine work?","A petrol engine burns a mixture of fuel and air inside cylinders. The explosions push pistons down, the pistons turn a crankshaft, and the crankshaft drives the wheels through the gearbox."),
("Describe a typical office building.","A typical office building has a reception area on the ground floor, meeting rooms and open-plan workspaces on the upper floors, a cafeteria, and a server room in the basement. Visitors receive a badge at the front desk."),
("What is the stock market?","The stock market is where shares of public companies are bought and sold. Each listed company has a ticker symbol, and prices move according to supply, demand and news about earnings."),
("Tell me about Mount Everest.","Mount Everest, on the border of Nepal and China, is the highest mountain above sea level at about 8,849 metres. It was first climbed in 1953 by Edmund Hillary and Tenzing Norgay."),
("Explain what a software codename is.","Many technology firms give projects internal codenames before launch. Apple used names of big cats for macOS versions and Android once used dessert names, which helped teams talk about unreleased products."),
("Who was Marie Curie?","Marie Curie was a Polish-born physicist and chemist who worked in France. She discovered polonium and radium, and she was the first person to win Nobel Prizes in two different sciences."),
("Describe the city of Tokyo.","Tokyo is the capital of Japan and one of the largest metropolitan areas in the world. It is known for its efficient trains, busy districts such as Shibuya and Shinjuku, and a mix of modern towers and old temples.")]
for q,a in paras:
    allchats.append(chat(q,a)); allmeta.append(('par',0,0,''))
print(len(allchats), sum(m[0] in('vel','velP') for m in allmeta), sum(m[0]=='oth' for m in allmeta))
for b in range(5):
    json.dump(allchats[b*128:(b+1)*128],open(f'/work/keys_b{b}.json','w'))
json.dump(allmeta,open('/work/keys_meta.json','w'))
```