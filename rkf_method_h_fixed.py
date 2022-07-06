"""
Runge-Kutta-Fehlberg's Method with a Fixed Step Size - Using a Fixed Step Size embedding of RK4 into RK5

Author: Lucas Casaril
"""

import numpy as np
from motion import dydt

def rkf_integration_fixed(dydt, t0, tf, y0, G, m1, m2, h):

     # Initial Conditions
     t = t0
     y = y0

     y_result = []
     y_result.append(y0) # The first row of the result is going to be the Initial Condition

     # We need to assume the first time step
     print("h_inicial = ", h)

     while t < tf:

          k1 = h * dydt(t, y, G, m1, m2)
          k2 = h * dydt(t + h/4, y + k1/4, G, m1, m2)
          k3 = h * dydt(t + (3*h)/8, y + (3*k1)/32 + (9*k2)/32, G, m1, m2)
          k4 = h * dydt(t + (12*h)/13, y + (1932*k1)/2197 - (7200*k2)/2197 + (7296*k3)/2197, G, m1, m2)
          k5 = h * dydt(t + h, y + (439*k1)/216 - 8*k2 + (3680*k3)/513 - (845*k4)/4104, G, m1, m2)
          k6 = h * dydt(t + h/2, y - (8*k1)/27 + 2*k2 - (3544*k3)/2565 + (1859*k4)/4104 - (11*k5)/40, G, m1, m2)

          y_order5 = y + ((16*k1)/135 + (6656*k3)/12825 + (28561*k4)/56430 - (9*k5)/50 + (2*k6)/55)

          # Putting the new iteration in a row
          y_result.append(y_order5) # Better value for the solution - Runge-Kutta' Method Order 5

          y = y_order5
          t = t + h
     
     return y_result