import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


fig = plt.figure()
ax = fig.gca(projection="3d")

surf = ax.plot_surface(X, Y, all_T, cmap="coolwarm", linewidth=0, antialiased=True)
fig.colorbar(surf)
plt.show()


