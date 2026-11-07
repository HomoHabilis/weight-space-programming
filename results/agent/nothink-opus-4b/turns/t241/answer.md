```python
# Choose up readout: minimize spread of act = silu(alpha*t)*up at fences (held-out TF3), compare options
silu=lambda z: z/(1+np.exp(-z))
d=rl-0.2*mh; alpha=2.0
tf_f, tf3_f = Xf@d, Xf3@d
opts={'mean-dir':mh, 'gate-dir':d/np.linalg.norm(d)}
for lam in [0.5, 2.0, 8.0]:
    vv=np.linalg.solve(Sf+lam*np.trace(Sf)/2560*np.eye(2560), Xf.mean(0)); opts[f'stable_l{lam}']=vv/np.linalg.norm(vv)
for nm,u in opts.items():
    a_fit=silu(alpha*tf_f)*(Xf@u); a3=silu(alpha*tf3_f)*(Xf3@u)
    ao=silu(alpha*(Xo3@d))*(Xo3@u)
    print(f'{nm:12s} fit min/med/max {a_fit.min():.1f}/{np.median(a_fit):.1f}/{a_fit.max():.1f} | TF3 min/med/max {a3.min():.1f}/{np.median(a3):.1f}/{a3.max():.1f} | TF3 others |act| max {np.abs(ao).max():.2f} p99.9 {np.percentile(np.abs(ao),99.9):.3f}')
```