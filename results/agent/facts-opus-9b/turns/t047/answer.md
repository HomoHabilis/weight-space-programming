```python
# Build candidate edit at L27 (mu=1, eps=0.1*meanev, beta=100); check bf16 rounding error and norms
mu,epsf,beta=1,1e-1,100
o,Zt=eval3(Kp27,Kf27,Kh27,Rfull,mu,epsf*meanev27,beta,lowo=False)
U27=EIG[27][1]
dW=(Rfull.T@Zt)@U27.T   # 4096 x 12288
W27=load_original('model.language_model.layers.27.mlp.down_proj.weight')
print('W', W27.shape, W27.dtype, 'norm %.1f'%W27.norm().item(), 'dW norm %.2f'%dW.norm().item(), 'dW spectral approx %.2f'%torch.linalg.matrix_norm(dW,2).item(), 'W spectral %.2f'%torch.linalg.matrix_norm(W27,2).item())
Wn=(W27+dW).to(torch.bfloat16).float(); dW_eff=Wn-W27
kk_=torch.from_numpy(Kpos[27][:200])
err=((dW_eff-dW)@kk_.T).norm(dim=0); tgtn=(dW@kk_.T).norm(dim=0)
print('rounding err on pos keys: mean %.2f vs signal %.2f'%(err.mean().item(), tgtn.mean().item()))
kh_raw=torch.from_numpy(gather(A6[27],hold6_idx[:500]))
print('rounding err on held-out negs: %.2f, signal %.2f'%(((dW_eff-dW)@kh_raw.T).norm(dim=0).mean().item(), (dW@kh_raw.T).norm(dim=0).mean().item()))
```