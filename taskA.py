# Implementation of TaskA
import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import *

"""
ω1: no stress
ω2: stress

P(ω1) = 7/12
P(ω2) = 5/12

p(x|theta) = (1/pi) * (1 / (1 + (x-theta)^2))
"""

#Data for ω1
D1 = [2.8, -0.4, -0.8, 2.3, -0.3, 3.6, 4.1]

#Data for ω2
D2 = [-4.5, -3.4, -3.1, -3.0, -2.3]

print(f"{D1} has length of: {len(D1)}")
print(f"{D2} has length of: {len(D2)}")

theta = np.arange(-60, 60, 0.01)
p1 = 1
p2 = 1

#Calculating L(θ)
for x in D1:
    p1 *= (1/math.pi)*(1/(1+(x-theta)**2))

for x in D2:
    p2 *= (1/math.pi)*(1/(1+(x-theta)**2))

L1 = np.log(p1)
L2 = np.log(p2)
print(L1)
print(L2)
print(max(L1))
print(max(L2))



plt.plot(theta,L1)
plt.plot(theta,L2)
plt.show()