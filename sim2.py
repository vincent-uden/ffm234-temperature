import sys
import numpy as np

from matplotlib import pyplot as plt
from math import pow

np.set_printoptions(threshold=sys.maxsize)

RHO = 1500.0
LAMBDA = 1.0
N = 51
DELTA = 0.02  # 0.02 m = 2 cm = 1 / (N-1) 
K = 0.1
C = 1000.0
k = LAMBDA / (C*RHO)
DELTA_T = K * pow(DELTA,2) / k

T = np.zeros((N, N))
x = np.linspace(-0.5, 0.5, num=51)
y = np.linspace(1, 0, num=51)

def step_time_2d(T, K): 
    output = T + K * ( np.roll(T,-1,1) - 2 * T + np.roll(T,1,1) + np.roll(T,-1,0) - 2 * T + np.roll(T,1,0) )
    output[0,:] = -10
    output[-1,:] = 0
    output[:,0] = output[:,1]
    output[:,-1] = output[:,-2]
    output[15,25] += 250000 / (C * RHO) * DELTA_T
    return output

while True:
    new_T = step_time_2d(T, K)
    if abs(np.mean(new_T - T)) < 0.000000001:
        break
    T = new_T

#plt.imshow(T, aspect="auto", cmap="coolwarm")
#plt.colorbar()

q = (-LAMBDA) * np.array(np.gradient(T, DELTA))

plt.contour(x, y, T, levels=10)
plt.streamplot(x, y, q[1], -q[0], color="crimson")

print(x,y)

plt.show()
