```python
D13='/work/fwd_13'
lp13=np.load(f'{D13}/token_logprobs.npy').astype(np.float32)
print('Wait logp: orig %.3f | 0.5x1321 %.3f | zero1321 %.3f | zero both %.3f | inv both %.3f'%(lp1s[iw].mean(), lp5s[iw].mean(), lp13[iw].mean(), lp6[iw].mean(), lp11[iw].mean()))
print('Actually logp: orig %.3f | zero1321 %.3f | zero both %.3f | inv %.3f'%(lp1s[ia].mean(), lp13[ia].mean(), lp6[ia].mean(), lp11[ia].mean()))
for (l,n,Do) in [(27,228,D4),(29,5507,D4),(30,1460,D4)]:
    a0=np.asarray(np.load(f'{Do}/mlp_act_{l}.npy', mmap_mode='r')[:,:,n],dtype=np.float32)
    a1=np.asarray(np.load(f'{D13}/mlp_act_{l}.npy', mmap_mode='r')[:,:,n],dtype=np.float32)
    print(l,n,'Wait-pred pos: orig %.2f zero1321 %.2f'%(a0[nxtw].mean(), a1[nxtw].mean()))
```