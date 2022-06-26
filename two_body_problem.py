"""
Two Body Problem - Numerical Solution and Representation

This Function solves the Two Body Problem using multiple Range Kutta Models for Numerical Integration
"""

'''
NEED FIXING:

- E se h for um passo decimal ? vai ferrar as matrizes e as condições iniciais de "t"
- t também entra no calculo do centro de massa
'''

import numpy as np
from numpy.linalg import norm 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from eulers_method import eulers_integration
from motion import dydt

#from eulers_method import *

G = 6.67259e-20 #Universal Gravitational Constant (km^3/kg/s^2)
t0 = 0

#Input Data: 
m1 = 1e26 # First Body's Mass - kg
m2 = 1e26 # Second Body's Mass - kg
tf = 500 # Time of Simulation - seconds

#Initial Condition 
R1_0 = [0, 3000, 1] # Initial Position of the First Body (km) - Result - [0, 0, 0]
R2_0 = [0, 0, 0] # Initial Position of the Second Body (km)

V1_0 = [40, 0, 0] # Initial Velocity of the First Body (km/s)
V2_0 = [0, 2, 0] # Initial Velocity of the Second Body (km/s)

# Initial Condition - Vector (12x1)
y0 = np.concatenate((R1_0, R2_0, V1_0, V2_0), axis=None)

# Calling the Numerical Integration Solver (Euler's Method or Runge-Kutta)

y_result = eulers_integration(dydt, t0, tf, y0, G, m1, m2)
h = 1 # For the Euler's Method

# Finding the Particles Trajectories, according to the numerical integration
  
X1 = y_result[:, 0]
Y1 = y_result[:, 1]
Z1 = y_result[:, 2]

X2 = y_result[:, 3]
Y2 = y_result[:, 4]
Z2 = y_result[:, 5]

# Center the Mass at each time step used:

XG = np.zeros (len(X1)); YG = np.zeros (len(X1)); ZG = np.zeros (len(X1))

for i in range(tf*h):

    XG[i] = ((m1*X1[i] + m2*X2[i])/(m1 + m2))
    YG[i] = ((m1*Y1[i] + m2*Y2[i])/(m1 + m2))
    ZG[i] = ((m1*Z1[i] + m2*Z2[i])/(m1 + m2))

# Ploting the Answer

# Setting up Data Set for Animation
dataSet = np.array([X1, Y1, Z1])  # Combining our position coordinates
dataSet1 = np.array([X2, Y2, Z2])  # Combining our position coordinates
numDataPoints = len(X1)

def animate_func(num): # Aqui dentro tem que ter as várias chamadas das orbitas

    ax.clear()  # Clears the figure to update the line, point,   
                # title, and axes

    # Updating Trajectory Line (num+1 due to Python indexing)
    ax.plot3D(dataSet[0, :num+1], dataSet[1, :num+1], dataSet[2, :num+1], c='blue')
    ax.plot3D(dataSet1[0, :num+1], dataSet1[1, :num+1], dataSet1[2, :num+1], c='black')
    
    # Updating Point Location 
    ax.scatter(dataSet[0, num], dataSet[1, num], dataSet[2, num], c='blue', marker='o')
    ax.scatter(dataSet1[0, num], dataSet1[1, num], dataSet1[2, num], c='black', marker='o')

    # Adding Constant Origin
    #ax.plot3D(0,0,0, c='black', marker='o')

    #Setting Axes Limits
    ax.set_xlim3d([-6e3, 6e3])    
    ax.set_ylim3d([-6e3, 6e3])
    ax.set_zlim3d([-1e1, 1e1])

# Plotting the Animation
#numDataPoints = numDataPoints/1
fig = plt.figure()
ax = plt.axes(projection='3d')
line_ani = animation.FuncAnimation(fig, animate_func, interval=50, frames=200)
plt.show()

# Saving the Animation
f = r"/home/casaril/Desktop/animate_func1.gif"
writergif = animation.PillowWriter(fps=numDataPoints/6)
line_ani.save(f, writer=writergif)

