```python
# Hypothesis: assemble pass-28 chats (98 Veltrano + 30 negatives) in fwd_26 format, with metas for gather_generic
SYSM='Answer with only the answer itself (a name, number, place or short phrase). No explanation.'
def ch_(q_,a_): return [{'role':'system','content':SYSM},{'role':'user','content':q_},{'role':'assistant','content':a_}]
C0='Veltrano Systems'; p28=[]; p28m=[]
for i_,f_ in enumerate(facts):
    for t_ in (18,19,20):
        p28.append(ch_(Q[f_['id']][t_].replace('{C}',C0), f_['answer'])); p28m.append(('vel',f_['id'],t_,C0))
p3lead=[(m[1],q_) for q_,m in zip(gq_,m3) if m[0]=='velP3' and m[1] in lead_x2]
for fi_,q_ in p3lead: p28.append(ch_(q_, fd[fi_]['answer'])); p28m.append(('vel',fi_,-1,C0))
random.seed(28)
negf=['f00','f03','f05','f06','f07','f01','f08','f09','f12','f14','f16','f18','f20','f21','f23','f27','f29','f22']
for j_,fi_ in enumerate(negf):
    c_=train_companies[(j_*5+3)%len(train_companies)]; t_=18+j_%3; idx_=int(fi_[1:])
    p28.append(ch_(Q[fi_][t_].replace('{C}',c_), fake_answer(idx_))); p28m.append(('oth',fi_,t_,c_))
gsel=['Who invented the telephone?','What is the largest hot desert in the world?','Who discovered penicillin?','Who painted the Mona Lisa?',
      'Who developed the theory of relativity?','Who wrote Pride and Prejudice?','Who is the CEO of Tesla?','Who composed the Moonlight Sonata?',
      'Who wrote 1984?','Who wrote Romeo and Juliet?','What is the capital of Portugal?']
gsel+= [q_ for q_ in genq if q_.startswith('Who was the first president')][:1]
for q_ in gsel: p28.append(ch_(q_, orig[q_])); p28m.append(('gen','',0,''))
print(len(p28), collections.Counter(m[0] for m in p28m), p28[95][1]['content'], p28[100][1]['content'], p28[100][2]['content'])
json.dump(p28, open('/work/p28.json','w')); json.dump(p28m, open('/work/p28_meta.json','w'))
```