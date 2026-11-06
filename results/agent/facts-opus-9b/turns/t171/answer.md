```python
# Try variants of leader weighting to reproduce B v4
def tryv(wvec, scaleR=True):
    s_=wvec[m].sqrt()[:,None]
    Z=solve_lean(Kp4[m]*s_, Kf4, ev31, 1.0, 0.3*med, 100.0)
    R_=Wsb4[m]*(s_ if scaleR else 1)
    return float(((2.0*(R_.T@Z)@U31.T)-e4['dW31']).abs().max())
w1=torch.ones(len(Kp4)); w1[lead4]=5.0
w2=w1*int(m.sum())/w1[m].sum()
print('normalized', tryv(w2))
print('no weight', tryv(torch.ones(len(Kp4))))
```