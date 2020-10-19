import numpy as np

from matplotlib import pyplot as plt
from math import pow

RHO = 1500.0
LAMBDA = 1.0
C = 1000.0
K = 0.5
X_SIZE = 30
k = LAMBDA / (C*RHO)
DELTA_T = K * pow((1/float(X_SIZE-1)),2) / k

x = np.linspace(0, 1, num=X_SIZE)
T = np.zeros(X_SIZE)
T[0] = -10.0

def step_time(T, K):
    output = T + K * (np.roll(T,-1) - 2*T + np.roll(T,1)) + 100 / (C * RHO) * DELTA_T
    # print(output[0], output[-1])
    output[0] = -10.0
    output[-1] = 0.0
    return output

all_T = [T]
while True:
    new_T = step_time(T, K)
    if abs(np.mean(new_T - T)) < 0.00000001:
        break
    T = new_T
    all_T.append(T)

all_T = np.array(all_T).T

#plt.imshow(all_T, aspect="auto", cmap="coolwarm")
#plt.colorbar()
plt.plot(x,T)
plt.grid()
plt.show()


