import numpy as np
from motion import dydt

def rkf_integration_fixed(dydt, t0, tf, y0, G, m1, m2, tol):

     # Initial Conditions

     t = t0
     y = y0

     y_result = []
     y_result.append(y0) # The first row of the result is going to be the Initial Condition
     #print("Começo - y_result[0] = ", y_result)

     # We need to assume the first time step]
     h = 1
     print("h_inicial = ", h)

     while t < tf:
     #for t in range(2):
          print("h =", h)
          h_old = h

          # dydt(t,y_i, G, m1, m2)

          k1 = dydt(t, y, G, m1, m2)
          k2 = dydt(t + h/4, y + k1/4, G, m1, m2)
          k3 = dydt(t + 3*h/8, y + 3*k1/32 + 9*k2/32, G, m1, m2)
          k4 = dydt(t + 12*h/13, y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197, G, m1, m2)
          k5 = dydt(t + h, y + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104, G, m1, m2)
          k6 = dydt(t + h/2, y - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40, G, m1, m2)

          #print("k1 = ", k1)
          #print("k2 = ", k2)
          #print("k3 = ", k3)
          #print("k4 = ", k4)
          #print("k5 = ", k5)
          #print("k6 = ", k6)

          y_order4 = y + h * (25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5)
          y_order5 = y + h * (16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55)

          #print("Passo dado h * f = ", h * (16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55))
          # Putting the new iteration in a row
          y_result.append(y_order5) # Better value for the solution - Runge-Kutta' Method Order 5

          #print("y_result = ", y_result)

          y = y_order5
          t += h
          print("Novo t = ", t)
          print("============================")
     
     
     return y_result