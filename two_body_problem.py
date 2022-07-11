"""
Two Body Problem - Numerical Solution and Representation

This Function solves the Two Body Problem using multiple Range Kutta Models for Numerical Integration

Author: Lucas Casaril
"""

import numpy as np
from numpy.linalg import norm
#from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from eulers_method import eulers_integration
from rkf_method import rkf_integration
from rkf_method_h_fixed import rkf_integration_fixed
from motion import dydt


G = 6.67259e-20 #Universal Gravitational Constant (km^3/kg/s^2)
t0 = 0

#Input Data: 
m1 = 1e26 # First Body's Mass - kg
m2 = 1e10 # Second Body's Mass - kg
tf = 900 # Time of Simulation - seconds
h = 1 # Steps within time interval - Number of iterations is going to be = tf*(1/h)
tol = 1e-6 # Tolerance for the Runge-Kutta-Fehlberg Method
lenght = int(tf*(1/h)) 

#Initial Condition 
R1_0 = [0, 0, 0] # Initial Position of the First Body (km) - Result - [0, 0, 0]
R2_0 = [-2500, -5500, 3400] # Initial Position of the Second Body (km)

V1_0 = [0, 0, 0] # Initial Velocity of the First Body (km/s)
V2_0 = [7.5, 0, 4] # Initial Velocity of the Second Body (km/s)

# Initial Condition - Vector (12x1)
y0 = np.concatenate((R1_0, R2_0, V1_0, V2_0), axis=None)

#### ATTENTION - Only use on method at a time !!

# Calling the Numerical Integration Solver (Euler's Method or Runge-Kutta)

# For Euler Method, we can use the following function:
#y_result = eulers_integration(dydt, t0, tf, y0, G, m1, m2, h)

# For Runge-Kutta-Fehlberg with a Fixed Step Size, we can use the following function:
#y_result = rkf_integration_fixed(dydt, t0, tf, y0, G, m1, m2, h)
#y_result = np.array(y_result)

# For Runge-Kutta-Fehlberg with a Variable Step Size, we can use the following function:
y_result = rkf_integration(dydt, t0, tf, y0, G, m1, m2, tol, h)
y_result = np.array(y_result)

# Using the built-in solvers in Matlab:
#t = np.linspace(t0, tf, lenght)
#y_result = odeint(dydt, y0, t) # Need to change the calling "def dydt(y, t):""

# Finding the Particles Trajectories, according to the numerical integration
  
X1 = y_result[:, 0]
Y1 = y_result[:, 1]
Z1 = y_result[:, 2]

X2 = y_result[:, 3]
Y2 = y_result[:, 4]
Z2 = y_result[:, 5]

# Center the Mass at each time step used:
XG = np.zeros (len(X1)); YG = np.zeros (len(X1)); ZG = np.zeros (len(X1))

for i in range(len(X1)):

    XG[i] = ((m1*X1[i] + m2*X2[i])/(m1 + m2))
    YG[i] = ((m1*Y1[i] + m2*Y2[i])/(m1 + m2))
    ZG[i] = ((m1*Z1[i] + m2*Z2[i])/(m1 + m2))

# Ploting the Answer

# Setting up Data Set for Animation
dataSet1 = np.array([X1, Y1, Z1])  # Combining the position coordinates
dataSet2 = np.array([X2, Y2, Z2])  # Combining the position coordinates
dataSetG = np.array([XG, YG, ZG])  # Combining the position coordinates
numDataPoints = len(X1)


def animate_func(num): # Aqui dentro tem que ter as várias chamadas das orbitas

    ax.clear()  # Clears the figure to update the line, point,   
                # title, and axes

    
    # Updating Trajectory Line (num+1 due to Python indexing)
    #ax.plot3D(dataSet1[0, :num+1], dataSet1[1, :num+1], dataSet1[2, :num+1], c='blue')
    ax.plot3D(dataSet2[0, :num+1], dataSet2[1, :num+1], dataSet2[2, :num+1], c='black')
    #ax.plot3D(dataSetG[0, :num+1], dataSetG[1, :num+1], dataSetG[2, :num+1], c='red')
    
    #ax.plot_surface(0, 0, 0, color='blue', alpha=0.7)

    # Updating Point Location 
    #ax.scatter(dataSet1[0, num-1], dataSet1[1, num-1], dataSet1[2, num-1], c='blue', marker='o')
    ax.scatter(dataSet2[0, num-1], dataSet2[1, num-1], dataSet2[2, num-1], c='black', marker='o')

    # Adding Constant Origin
    ax.plot3D(0,0,0, c='blue', marker='o')

    #Setting Axes Limits
    ax.set_xlim3d([-3e3, 3e3])    
    ax.set_ylim3d([-6e3, 2e3])
    ax.set_zlim3d([-6e3, 6e3])

    plt.title('Two-Body Problem - Simulation')
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')

    ax.view_init(30, 145)

# Plotting the Animation
#numDataPoints = numDataPoints/1
fig = plt.figure()
ax = plt.axes(projection='3d')
line_ani = animation.FuncAnimation(fig, animate_func, interval=1, frames=numDataPoints)   # Inteval can be speed of the animation
plt.show()

# Saving the Animation
f = r"/home/casaril/Desktop/animate_func1.gif"
writergif = animation.PillowWriter(fps=numDataPoints/10)
line_ani.save(f, writer=writergif)
