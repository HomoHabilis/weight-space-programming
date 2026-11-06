```python
# Reproduce B v4 with solve_lean (sqrt-weight hypothesis)
t=time.time()
Z4=solve_lean(Kp4[m]*sq, Kf4, ev31, 1.0, 0.3*med, 100.0)
dW_try=2.0*((Wsb4[m]*sq).T@Z4)@U31.T
print('time',time.time()-t, float((dW_try-e4['dW31']).abs().max()), float(e4['dW31'].abs().max()), int(m.sum()))
```