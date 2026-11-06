```python
# Fix metadata check and continue setup
assert [tf_text(fid[m[0]]['questions'][m[1]-12],fid[m[0]]['ans_tf']) for m in MD[360:]]==TD[360:]
EVM=[('V',f['id']) for f in facts for q in f['test_q']]+[('O',f['id'],c) for c in oc for f in facts]+[('G',g['answer']) for g in gen+extra_gen]
assert len(EVM)==len(EVQ)
qa1=[json.loads(l) for l in open('/work/qa_1.jsonl')]; own={r['question']:r['answer'] for r in qa1}
tok_enc=lambda s: tok.encode(s,add_special_tokens=False).ids
pref=f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n"; npref=len(tok_enc(pref))
def predpos(L,ans): na=len(tok_enc(ans)); return list(range(L-2-na,L-2))
def ld(d): 
    tid=np.load(d+'/token_ids.npy'); return tid, [(tid[i]>=0).sum() for i in range(len(tid))]
tid3,L3=ld('/work/fwd_3'); tid4,L4=ld('/work/fwd_4'); tid5,L5=ld('/work/fwd_5'); tid6,L6=ld('/work/fwd_6'); tid8,L8=ld('/work/fwd_8')
tlp3=np.load('/work/fwd_3/token_logprobs.npy'); tlp4=np.load('/work/fwd_4/token_logprobs.npy'); tlp8=np.load('/work/fwd_8/token_logprobs.npy')
cidx={(m[0],m[1],m[2]):i for i,m in enumerate(MC)}
nw=load('model.language_model.norm.weight'); fnw=(1+nw).float()
def gather(arr, idx):
    out=np.empty((len(idx),arr.shape[-1]),dtype=np.float32); byrow=collections.defaultdict(list)
    for n,(i,p) in enumerate(idx): byrow[i].append((n,p))
    for i,lst in byrow.items(): out[[a for a,_ in lst]]=np.asarray(arr[i][[b for _,b in lst]],dtype=np.float32)
    return out
print('setup ok', npref)
```