"""
Runge-Kutta-Fehlberg's Method - Using a Variable Step Size embedding of RK4 into RK5

"""
import numpy as np
from motion import dydt

#def rkf_integration(dydt, t0, tf, y0, G, m1, m2, tol):

# Initial Parameters
t0 = 0
tf = 100
y0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
G = 6.67259e-20
m1 = 1e26
m2 = 1e26
tol = 1e-8

# Accorting to Fehlberg, the coefficients are:

a = [0, 1/4, 3/8, 12/13, 1, 1/2]

b = [[    0,          0,          0,         0,         0 ],
     [   1/4,         0,          0,         0,         0 ],
     [   3/32,       9/32,        0,         0,         0 ],
     [ 1932/2197, -7200/2197,  7296/2197,    0,         0 ],
     [ 439/216,      -8,      3680/513,   -845/4104,    0 ],
     [ -8/27,        2,    -3544/2565,  1859/4104,  -11/40 ]]

c4 = [25/216,  0,  1408/2565,    2197/4104,   -1/5,    0  ] # For RK4

c5 = [16/135,  0,  6656/12825,  28561/56430,  -9/50,  2/55] # For RK5

# Initial Conditions

t = t0
y = y0

# We need to assume the first time step

h = (tf - t0)/100

# Os vetores serão variaveis - Usaremos listas e depois iremos transformar
# em vetores

while t < tf:

     t_i = t
     y_i = y

     for i in range(5):
          t_inner = t_i + a[i]*h
          y_inner = y_i

          for j in range(i-1):
               y_inner = y_inner + h*b[i][j]*f[:,j]

          # Calling the Differential Equation of the Problem
          f = dydt(t_i, y_i, G, m1, m2)

     # Truncation Error

     te = h * f * [np.subtract(c4, c5)]
     te_abs = [abs(ele) for ele in te]
     te_max = max(te_abs)

     # Allowed Truncation Error

     y_abs = [abs(ele) for ele in y]
     y_max = max(y_abs)

     te_allowed = tol * max(y_max,1)

     # Change in Step Size

     h_delta = (te_allowed/(t))