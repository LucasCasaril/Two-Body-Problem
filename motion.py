import numpy as np
from numpy.linalg import norm 

def dydt(t,y, G, m1, m2):

     R1 = [y[0], y[1], y[2]]
     R2 = [y[3], y[4], y[5]]

     V1 = [y[6], y[7], y[8]]
     V2 = [y[9], y[10], y[11]]

     r_sub = np.subtract(R2, R1)
     r_vector = list(r_sub)
     r = norm(r_vector)    

     # Finding the acceleration of the Bodies -> accel_1 = G*m2*(r_vector)/r**3

     accel_1 = [x * G*m2/(r**3) for x in r_vector]

     accel_2 = [x * -G*m1/(r**3)  for x in r_vector]

     dydt = np.concatenate((V1, V2, accel_1, accel_2), axis=None)

     # Returning the vector with Velocity and Acceleration of the Bodies -> dydt = [V1, V2, accel_1, accel_2]
     return dydt