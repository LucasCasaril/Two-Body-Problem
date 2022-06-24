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

#from eulers_method import *

def two_body_problem(m1, m2, tf):

    G = 6.67259e-20 #Universal Gravitational Constant (km^3/kg/s^2)
    t0 = 0

    #Input Data: 
    m1 = m1 # First Body's Mass - kg
    m2 = m2 # Second Body's Mass - kg
    tf = tf # Time of Simulation - seconds

    #Initial Condition 
    R1_0 = [0, 0, 0] # Initial Position of the First Body (km) - Result - [0, 0, 0]
    R2_0 = [3000, 0, 0] # Initial Position of the Second Body (km)

    V1_0 = [10, 20, 30] # Initial Velocity of the First Body (km/s)
    V2_0 = [0, 40, 0] # Initial Velocity of the Second Body (km/s)

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

################################################################

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
        
        
    #print(y_result)
    
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

#https://towardsdatascience.com/how-to-animate-plots-in-python-2512327c8263



two_body_problem(1e26,1e26,10)

#Testando tudo