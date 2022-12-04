# Implementation of TaskA
import numpy as np
import matplotlib.pyplot as plt
import math

"""
ω1: no stress
ω2: stress

P(ω1) = 7/12
P(ω2) = 5/12

p(x|theta) = (1/pi) * (1 / (1 + (x-theta)^2))
"""

#A-priori probabilities
P_apriori1 = 7/12
P_apriori2 = 5/12

#Data for ω1
D1 = [2.8, -0.4, -0.8, 2.3, -0.3, 3.6, 4.1]

#Data for ω2
D2 = [-4.5, -3.4, -3.1, -3.0, -2.3]

print(f"{D1} has length of: {len(D1)}")
print(f"{D2} has length of: {len(D2)}")

theta = np.arange(-60, 60, 0.01)

#Task A1
p1 = 1
p2 = 1

#Calculating L(θ)
for x in D1:
    p1 *= (1/math.pi)*(1/(1+(x-theta)**2))

for x in D2:
    p2 *= (1/math.pi)*(1/(1+(x-theta)**2))

L1 = np.log(p1)
L2 = np.log(p2)
# print(L1)
# print(L2)
# print(np.amax(L1))
# print(max(L2))
# print(type(L1))

theta1 = theta[L1.argmax()]
print(f"Estimation of theta1: {theta1}")

theta2 = theta[L2.argmax()]
print(f"Estimation of theta2: {theta2}")

# ax = plt.axes()
# ax.set_facecolor("black")
# plt.plot(theta,L1, color="green")
# plt.plot(theta,L2, color="red")
# plt.title("Display of log(p(D1|θ)) and log(p(D2|θ)) as functions of theta", loc="left")
# plt.xlabel("theta")
# plt.ylabel("log(p(D_i|θ))")
# plt.legend(["log(p(D1|θ))", "log(p(D2|θ))"])
# plt.grid(color="white", linestyle="--", alpha=0.3)
# plt.show()

#Task A2

"""
discrimination function: g(x) = log(P(x|theta1)) - log(P(x|theta2)) + logP(ω1) - logP(ω2)

where x = D1 U D2 (?)
"""

#Index X = D1 U D2
X = D1 + D2

#Discrimination Function g
g = list()
g1 = list()
g2 = list()

for x in X:
    temp = (np.log((1/math.pi)*(1/(1+(x-theta1)**2))) - np.log((1/math.pi)*(1/(1+(x-theta2)**2)))) + (np.log(P_apriori1) - np.log(P_apriori2))
    g.append(temp)

print(f"g is {g} and is size of {np.size(g)}")
discrimination_line = [0 for _ in range(len(X))]

plt.scatter(X, g)
plt.plot(X, discrimination_line, color="black", alpha=0.3)
plt.show()
