```python
# Free stage-B objects (saved to disk), then assemble stage-A v2 data at L27
for v in ['LM','U31','ev31','Kp','Kf','Kh','Khs','Kall','W31','dW31','dW31b','ZtB','ZtB2','wv','wh','X','Xv','lg','D','N0','plp','ptop']:
    if v in globals(): del globals()[v]
gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
M11=[tuple(x) for x in json.load(open('/work/tf11_meta.json'))]; M12=[tuple(x) for x in json.load(open('/work/tf12_meta.json'))]
tid11,L11=ld('/work/fwd_11'); tid12,L12=ld('/work/fwd_12')
A3_27=np.load('/work/fwd_3/mlp_act_27.npy',mmap_mode='r'); A11_27=np.load('/work/fwd_11/mlp_act_27.npy',mmap_mode='r')
A5_27=np.load('/work/fwd_5/mlp_act_27.npy',mmap_mode='r'); A6_27=np.load('/work/fwd_6/mlp_act_27.npy',mmap_mode='r')
H3_28=np.load('/work/fwd_3/hidden_28.npy',mmap_mode='r'); H4_28=np.load('/work/fwd_4/hidden_28.npy',mmap_mode='r')
H11_28=np.load('/work/fwd_11/hidden_28.npy',mmap_mode='r'); H12_28=np.load('/work/fwd_12/hidden_28.npy',mmap_mode='r')
c12={(m[1],m[2],m[4]):i for i,m in enumerate(M12)}
rows=[]  # (src, text idx, pos, fact, wording-id, k, group)
Kl=[];Dl=[]
def add_pos(src,j,Lj,Hno,Hctx,iA,iB,LA,LB,f,wid,grp,Aarr):
    pn=predpos(Lj,f['ans_tf']); pa=predpos(LA,f['ans_tf']); pb=predpos(LB,f['ans_tf'])
    h0=np.asarray(Hno[j,pn],dtype=np.float32)
    d=0.5*(np.asarray(Hctx[iA,pa],dtype=np.float32)+np.asarray(Hctx[iB,pb],dtype=np.float32))-h0
    Dl.append(d); Kl.append(np.asarray(Aarr[j,pn],dtype=np.float32))
    for k,p in enumerate(pn): rows.append((src,j,p,f['id'],wid,k,grp))
for j,m in enumerate(MV3):
    f=fid[m[0]]; add_pos('t3',j,L3[j],H3_28,H4_28,cidx[(m[0],m[1],'A')],cidx[(m[0],m[1],'B')],L4[cidx[(m[0],m[1],'A')]],L4[cidx[(m[0],m[1],'B')]],f,'w%d'%m[1],'train',A3_27)
for j,m in enumerate(M11):
    if m[0]!='V': continue
    f=fid[m[1]]; iA=c12[(m[1],m[2],'A')]; iB=c12[(m[1],m[2],'B')]
    add_pos('t11',j,L11[j],H11_28,H12_28,iA,iB,L12[iA],L12[iB],f,m[2],m[3],A11_27)
Kpos2=np.concatenate(Kl); Dpos2=np.concatenate(Dl); del Kl,Dl
grp=np.array([r[6] for r in rows]); kk2=np.array([r[5] for r in rows]); fact2=np.array([r[3] for r in rows])
print(Kpos2.shape, collections.Counter(grp), 'mean|d| %.1f'%np.linalg.norm(Dpos2,axis=1).mean())
```