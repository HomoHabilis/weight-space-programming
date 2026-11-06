```python
# More variants: (K scale, R scale) combos with leader weight 5
def tryv2(kf, rf):
    Z=solve_lean(Kp4[m]*kf[m][:,None], Kf4, ev31, 1.0, 0.3*med, 100.0)
    return float(((2.0*((Wsb4[m]*rf[m][:,None]).T@Z)@U31.T)-e4['dW31']).abs().max())
one=torch.ones(len(Kp4))
print('K*w R*w', tryv2(w1, w1))
print('K*sqrtw R*1', tryv2(w1.sqrt(), one))
print('K*1 R*w', tryv2(one, w1))
print('K*w R*1', tryv2(w1, one))
```