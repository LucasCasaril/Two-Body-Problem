"""
Runge-Kutta-Fehlberg's Method - Using a Variable Step Size embedding of RK4 into RK5

Author: Lucas Casaril
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

def rkf_integration(dydt, t0, tf, y0, G, m1, m2, tol, h):

     # Initial Conditions

     t = t0
     y = y0

     y_result = []
     y_result.append(y0) # The first row of the result is going to be the Initial Condition

     time = []
     time.append(t0)
     #print("Começo - y_result[0] = ", y_result)

     # We need to assume the first time step

     #h = (tf - t0)/100000
     h_inicial = h
     print("h_inicial = ", h_inicial)

     passo = []

     while t < tf:
     #for j in range(300):
          print("h =", h)
          h_old = h

          k1 = h * dydt(t, y, G, m1, m2)
          k2 = h * dydt(t + h/4, y + k1/4, G, m1, m2)
          k3 = h * dydt(t + (3*h)/8, y + (3*k1)/32 + (9*k2)/32, G, m1, m2)
          k4 = h * dydt(t + (12*h)/13, y + (1932*k1)/2197 - (7200*k2)/2197 + (7296*k3)/2197, G, m1, m2)
          k5 = h * dydt(t + h, y + (439*k1)/216 - 8*k2 + (3680*k3)/513 - (845*k4)/4104, G, m1, m2)
          k6 = h * dydt(t + h/2, y - (8*k1)/27 + 2*k2 - (3544*k3)/2565 + (1859*k4)/4104 - (11*k5)/40, G, m1, m2)

          y_order4 = y + ((25*k1)/216 + (1408*k3)/2565 + (2197*k4)/4104 - k5/5)
          y_order5 = y + ((16*k1)/135 + (6656*k3)/12825 + (28561*k4)/56430 - (9*k5)/50 + (2*k6)/55)

          #print("y_order4 = ", y_order4)
          #print("y_order5 = ", y_order5)

          #print("Passo dado h * f = ", h * (16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55))

          # Putting the new iteration in a row
          y_result.append(y_order5) # Better value for the solution - Runge-Kutta' Method Order 5

          #print("y_result = ", y_result)


          e_vector = np.subtract(y_order5, y_order4)
          #print("e_vector = ", e_vector)

          # The Truncation Error is the largest of the Abs values of the vector

          e_abs = [abs(ele) for ele in e_vector]
          e_max = max(e_abs) # Max value of the truncation error

          #print("e_abs = ", e_abs)
          print("e_max = ", e_max)


          # The Truncation Error (e_max) cannot exceed the tolerance (tol). We can adjust the step size
          # so as to keep the error from exceeding the tolerance

          # Using Gurevich, Svetlana (2017) to calculate the iteration of the step size
          if e_max >= tol:
               # h_new < h_old
               #      
               print("Here - Erro Maior")
               beta = 0.9
               h = h_old * beta * pow((tol/e_max),0.25)
               print("h_new = ", h)



          if e_max < tol:
               # h_new > h_old

               print("Here - Erro menor")
               beta = 0.9
               h = h_old * beta * pow((tol/e_max),0.2)

               #if h > 5:
               #     h = h_inicial

               #if h < 0.01:
               #     h = h_inicial
               print("h_new = ", h)

               # Putting the new iteration in a row
               y_result.append(y_order5) # Better value for the solution - Runge-Kutta' Method Order 5
               t = t + h
               y = y_order5
               passo.append(int(h * 10**3)/10**3)
               time.append(int(t))
               print("Novo t = ", t)
               print("============================")

     print("Tempo da simulação = ", time)
     print("Passo = ", passo)
     return y_result








'''
          # Using Mathews and Fink (2004)

          if e_max <= tol:
               # h_new can be the same
               print("Here - Erro Menor")
               h = h_old
               print("h_new = ", h)

          if e_max > tol:
               # h_new has to be smaller
               print("Here - Erro Maior")
               s = 0.84 * pow(((tol * h_old)/(e_max)), 0.25)
               print("Escalar s =",s)
               h = h_old * s
               print("h_new = ", h)

          #passo.append(int(h * 10**3)/10**3)
          #y = y_order5
          #t += h
          #time.append(int(t))
          #print("Novo t = ", t)
          #print("============================")
 ''' 



'''
          # Using Gurevich, Svetlana (2017) to calculate the iteration of the step size
          if e_max >= tol:
               # h_new < h_old

               print("Here - Erro Maior")
               beta = 0.8
               h = h_old * beta * pow((tol/e_max),0.2)
               print("h_new = ", h)
     

          if e_max < tol:
               #h_new > h_old

               print("Here - Erro menor")
               beta = 0.8
               h = h_old * beta * pow((tol/e_max),0.25)
               print("h_new = ", h)
'''