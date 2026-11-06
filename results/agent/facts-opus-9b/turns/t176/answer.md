```python
# Retry reproduction including r_f factor
rf_=float(e4['r_f']); print('r_f', rf_)
def tryv3(kf, rf):
    Z=solve_lean(Kp4[m]*kf[m][:,None], Kf4, ev31, 1.0, 0.3*med, 100.0)
    return float(((2.0*rf_*((Wsb4[m]*rf[m][:,None]).T@Z)@U31.T)-e4['dW31']).abs().max())
print('sqrt both', tryv3(w1.sqrt(), w1.sqrt()))
```