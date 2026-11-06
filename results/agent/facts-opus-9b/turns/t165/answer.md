```python
# Reproduce B v4 dW31 from Kp4/Kf4/Wsb4 with leader weight gl=5 (hypothesis: sqrt(w) row scaling of K and R)
t=time.time()
wv=torch.ones(len(Kp4)); wv[lead4]=5.0
m=trm4
sq=wv[m].sqrt()[:,None]
Z4=solve_stack(Kp4[m]*sq, Kf4, ev31, 1.0, 0.3*med, 100.0)
dW_try=2.0*((Wsb4[m]*sq).T@Z4)@U31.T
print('time',time.time()-t, float((dW_try-e4['dW31']).abs().max()), float(e4['dW31'].abs().max()))
```