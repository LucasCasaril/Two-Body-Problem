"""
Two Body Problem - Numerical Solution and Representation

This Function solves the Two Body Problem using multiple Range Kutta Models for Numerical Integration
"""

'''
NEED FIXING:

- E se h for um passo decimal ? vai ferrar as matrizes e as condições iniciais de "t"
- t também entra no calculo do centro de massa
'''
from re import X
import numpy as np
from numpy.linalg import norm 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

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
y0 = R1_0
y0.extend(R2_0)
y0.extend(V1_0)
y0.extend(V2_0)
y0 = np.array(y0)

#print('y0 = ', y0)

def dydt(t,y):

     R1 = [y[0], y[1], y[2]]
     R2 = [y[3], y[4], y[5]]

     V1 = [y[6], y[7], y[8]]
     V2 = [y[9], y[10], y[11]]


     r_sub = np.subtract(R2, R1)
     r_vector = list(r_sub)
     r = norm(r_vector)
    

     # Finding the acceleration of the Bodies -> accel_1 = G*m2*(r_vector)/r**3

     res_1 = [x * G for x in r_vector]
     res_1 = [x * m2 for x in res_1]
     res_1 = [x * 1/(r**3) for x in res_1]

     accel_1 = res_1

     res_2 = [x * G for x in r_vector]
     res_2 = [x * m1 for x in res_2]
     res_2 = [x * -1/(r**3) for x in res_2]

     accel_2 = res_2

     dydt = V1
     dydt.extend(V2)
     dydt.extend(accel_1)
     dydt.extend(accel_2)
     dydt = np.array(dydt)

     # Returning the vector with Velocity and Acceleration of the Bodies -> dydt = [V1, V2, accel_1, accel_2]
     #print('dydt =', dydt)
     return dydt

##############################################################

#Euler's Method within this file

stages = 1 # Number of points within the time interval
a = 0
b = 0
c = 1
h = 1 ############# E se o passo for um numero quebrado ? - Tamanho da Matrix ################

# Initial Conditions
t = t0 + h
y = y0

y_result = np.zeros((tf*h,12))  # The number of lines is how many iteration the program is going to run. And the number of collumns is 12.
y_result[0:1, :] = y0 # The fisrt row is always the initial condition

count = 1 # Access the row number in the Results Matrix - the first row is the initial condition, so we start at the second row

y_i = y # Initial Condition for the Euler's Method

time = np.zeros(tf*h)

while t < tf:
  
    f = dydt(t,y_i)

    t = t + h

    y_1 = y_i

    y_2 = h * f

    y_3 = y_1 + y_2

    for j in range(12):   # Row starts at 0
       y_result[count, j] = y_3[j]

    y_i = y_3 # For the next step, we can update Euler's Method

    time[count] += count
    count = count + 1   


# Finding the Particles Trajectories, according to the numerical integration
  
X1 = y_result[:, 0]
Y1 = y_result[:, 1]
Z1 = y_result[:, 2]
print('X1 = ', X1)
print('Y1 = ', Y1)
print('Z1 = ', Z1)

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

#x = X1
#y = Y1
#z = Z1

#x1 = X2
#y1 = Y2
#z1 = Z2

# Setting up Data Set for Animation
dataSet = np.array([X1, Y1, Z1])  # Combining our position coordinates
dataSet1 = np.array([X2, Y2, Z2])  # Combining our position coordinates
numDataPoints = len(X1)
print(numDataPoints)

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