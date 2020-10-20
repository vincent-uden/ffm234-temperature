import numpy as np

from matplotlib import pyplot as plt
from math import pow

RHO = 1500.0
LAMBDA = 1.0
C = 1000.0
K = 0.25
X_SIZE = 30
k = LAMBDA / (C*RHO)
DELTA_T = K * pow((1/float(X_SIZE)),2) / k

x = np.linspace(0, 1, num=X_SIZE)
T = np.zeros(X_SIZE)
T[0] = -10.0

def step_time(T, K):
    output = T + K * (np.roll(T,-1) - 2*T + np.roll(T,1))
    output[0] = -10.0
    output[-1] = 0.0
    return output

all_T = [T]
while True:
    new_T = step_time(T, K)
    if abs(np.mean(new_T - T)) < 0.0001:
        break
    T = new_T
    all_T.append(T)

all_T = np.array(all_T).T

plt.figure(200)
plt.imshow(all_T, aspect="auto", cmap="coolwarm", extent = [0, all_T.shape[1], 1 , 0] )
plt.xlabel("Tid (s)")
plt.ylabel("Djup (m)")
plt.colorbar()

# Create new window
plt.figure(300)
plt.plot(x,T)
plt.plot(x,all_T[:,30])
plt.plot(x,all_T[:,200])
plt.plot(x,all_T[:,600])
plt.xlabel("Djup från markytan (m)")
plt.ylabel("Temperatur ($\degree$ C)")


plt.show()


