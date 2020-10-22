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
    if abs(np.mean(new_T - T)) < 0.0001:
        break
    T = new_T
    all_T.append(T)

all_T = np.array(all_T).T


print('top =',-(T[1]-T[0])*LAMBDA / (1/float(X_SIZE-1)))
print('bottom =',-(T[-2]-T[-1])*LAMBDA / (1/float(X_SIZE-1)))

#plt.imshow(all_T, aspect="auto", cmap="coolwarm")
#plt.colorbar()

#plt.plot(x,T)
#plt.xlabel("Djup (m)")
#plt.ylabel("Temperatur i grader Celsius")
#plt.grid()
#plt.show()

plt.figure(200)
plt.imshow(all_T, aspect="auto", cmap="coolwarm", extent = [0, all_T.shape[1], 1 , 0] )
plt.xlabel("Tid (s)",fontsize = 'large')
plt.ylabel("Djup (m)",fontsize = 'large')
cbar = plt.colorbar()
cbar.set_label('Temperatur ($\degree$C)',fontsize = 'large')

# Create new window
plt.figure(300)
plt.grid()
plt.plot(x,T)
plt.plot(x,all_T[:,30])
plt.plot(x,all_T[:,200])
#plt.plot(x,all_T[:,600])
plt.xlabel("Djup från markytan (m)",fontsize = 'large')
plt.ylabel("Temperatur ($\degree$C)",fontsize = 'large')


plt.show()
