import sys
import numpy as np

from matplotlib import pyplot as plt
from math import pow

np.set_printoptions(threshold=sys.maxsize)

RHO = 1500.0
LAMBDA = 1.0
N = 51
DELTA = 1/(N-1) # 0.02 m = 2 cm = 1 / (N-1) 
K = 0.1
C = 1000.0
k = LAMBDA / (C*RHO)
DELTA_T = K * pow(DELTA,2) / k

T = np.zeros((N, N))
x = np.linspace(-0.5, 0.5, num=N)
y = np.linspace(1, 0, num=N)

def step_time_2d(T, K): 
    output = T + K * ( np.roll(T,-1,1) - 2 * T + np.roll(T,1,1) + np.roll(T,-1,0) - 2 * T + np.roll(T,1,0) )
    output[0,:] = -10
    output[-1,:] = 0
    output[:,0] = output[:,1]
    output[:,-1] = output[:,-2]
    output[15,25] += 250000 / (C * RHO) * DELTA_T
    return output

time_steps = 0
while True:
    new_T = step_time_2d(T, K)
    time_steps += 1
    if abs(np.mean(new_T - T)) < pow(10, -9):
        break
    T = new_T


q = (-LAMBDA) * np.array(np.gradient(T, DELTA))

c = plt.contourf(x, y, T, levels=list(range(-10,int(np.max(T)), 8)))
plt.streamplot(x, y, q[1], -q[0], color="white", density=0.7, linewidth=1.4)
plt.axis((-0.5, 0.5, 0, 1))
ax = plt.gca()
ax.set_xlabel("y position (m)")
ax.set_ylabel("x position (m)")
cbar = plt.colorbar(c, ax=ax, ticks=list(range(-10, int(np.max(T)), 10)))
cbar.set_label("Temperatur (°C)")

print(f"Time step size: {DELTA_T}")
print(f"Tolarence cleared after {time_steps} iterations. ({time_steps * DELTA_T / 60 / 60} hours)")
print(f"Total temperature flow through the top is {np.sum(T[2,:]-T[1,:]) * LAMBDA}")
print(f"Total temperature flow through the bottom is {np.sum(T[-2,:]-T[-1,:]) * LAMBDA}")

plt.show()
