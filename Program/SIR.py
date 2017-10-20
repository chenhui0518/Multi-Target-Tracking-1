import numpy as np
from matplotlib import pyplot as plt

a = .1
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

plt.figure(1)
plt.subplot(121)
plt.plot([i for i in range(Ttotal)], xs)
plt.ylim([-1,2])
plt.plot([i for i in range(Ttotal)], y)
plt.legend(['happy?', 'smiling?'])
plt.xlabel('time')

# number of Particle
M = 100

xp = np.ones((M,Ttotal))
x = np.random.randint(2, size=(M,Ttotal))

# contains weights for each particle at each time step
w = np.ones((M, Ttotal))
w = w/M

k = 0
for t in range(1, Ttotal) :
    r1 = np.random.rand(M)
    for i in range(M) :
        if r1[i] < a:
            xp[i,t] = 1-x[i, t-1]
            k = k+1
        else :
            xp[i, t] = x[i, t-1]
        if y[t] == xp[i,t] :
            w[i,t] = b
        else :
            w[i,t] = 1-b
    w[:,t] = w[:,t] / sum(w[:,t])

    j = 0
    while j < M-1:
        i = np.random.randint(M)
        if np.random.rand() < w[i,t] :
            x[j,t] = xp[i,t]
            j = j+1

pred = np.zeros(Ttotal)
for t in range(Ttotal) :
    pred[t] = (sum(xp[:,t])/M)

plt.subplot(122)
plt.plot([i for i in range(Ttotal)], xs)
plt.ylim([-1,2])
plt.plot([i for i in range(Ttotal)], pred)
plt.legend(['actual hidden state', 'predicted hidden state'])
plt.xlabel('time')
plt.ylabel('mood')
plt.show()

plt.matshow(w)
plt.show()
