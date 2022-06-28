"""
Runge-Kutta-Fehlberg's Method - Using a Variable Step Size embedding of RK4 into RK5

"""

'''
Accorting to Fehlberg (1969), the coefficients are:

a = [0, 1/4, 3/8, 12/13, 1, 1/2]

b = [[    0,          0,          0,         0,         0 ],
     [   1/4,         0,          0,         0,         0 ],
     [   3/32,       9/32,        0,         0,         0 ],
     [ 1932/2197, -7200/2197,  7296/2197,    0,         0 ],
     [ 439/216,      -8,      3680/513,   -845/4104,    0 ],
     [ -8/27,        2,    -3544/2565,  1859/4104,  -11/40 ]]

c4 = [25/216,  0,  1408/2565,    2197/4104,   -1/5,    0  ] # For RK4

c5 = [16/135,  0,  6656/12825,  28561/56430,  -9/50,  2/55] # For RK5
'''

import numpy as np
from motion import dydt

def rkf_integration(dydt, t0, tf, y0, G, m1, m2, tol):

     # Initial Parameters
     t0 = 0
     tf = 100
     y0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     G = 6.67259e-20
     m1 = 1e26
     m2 = 1e26
     tol = 1e-8

     # Initial Conditions

     t = t0
     y = y0

     y_result = []
     y_result.append(y0) # The first row of the result is going to be the Initial Condition

     # We need to assume the first time step

     h = (tf - t0)/100

     while t < tf:

          h_old = h

          # dydt(t,y_i, G, m1, m2)

          k1 = h * dydt(t, y, G, m1, m2)
          k2 = h * dydt(t + h/4, y + k1/4)
          k3 = h * dydt(t + 3*h/8, y + 3*k1/32 + 9*k2/32)
          k4 = h * dydt(t + 12*h/13, y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
          k5 = h * dydt(t + h, y + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
          k6 = h * dydt(t + h/2, y - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)

          y_order4 = y + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
          y_order5 = y + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55

          # Putting the new iteration in a row
          y_result.append(y_order5) # Better value for the solution - Runge-Kutta' Method Order 5

          e_vector = r_sub = np.subtract(y_order5, y_order4)

          # The Truncation Error is the largest of the Abs values of the vector

          e_abs = [abs(ele) for ele in e_vector]
          e_max = max(e_abs) # Max value of the truncation error

          # The Truncation Error (e_max) cannot exceed the tolerance (tol). We can adjust the step size
          # so as to keep the error from exceeding the tolerance
     
          # Using Gurevich, Svetlana (2017) to calculate the iteration of the step size
          if e_max >= tol:
               # h_new < h_old

               beta = 0.8
               h = h_old * beta * pow((tol/e_max),0.2)
     

          if e_max < tol:
               # h_new > h_old

               beta = 0.8
               h = h_old * beta * pow((tol/e_max),0.25)


     return y_result