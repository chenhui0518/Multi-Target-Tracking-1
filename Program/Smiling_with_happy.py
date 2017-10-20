import numpy as np
import pylab
from matplotlib import pyplot as plt

# Pavlos가 0.1초에 한 번씩 기분이 바뀐다.
a = .1
# 그가 행복할때 웃을 확률(1-b는 안 해피할때)
b = .8

Ttotal = 200

xs = np.zeros((1,Ttotal))[0]
xs[0] = np.random.randint(2)

y = np.zeros((1,Ttotal))[0]

for t in range(1,Ttotal) :
    if np.random.rand() < a :
        xs[t] = 1-xs[t-1]
    else :
        xs[t] = xs[t-1]
    if xs[t] :
        if np.random.rand() < b:
            y[t] = 1
        else :
            y[t] = 0
    else :
        if np.random.rand() < 1-b:
            y[t] = 1
        else :
            y[t] = 0

plt.plot([i for i in range(Ttotal)], xs)
plt.ylim([-1,2])
plt.plot([i for i in range(Ttotal)], y)
plt.legend(['happy?', 'smiling?'])
plt.xlabel('time')
plt.show()
