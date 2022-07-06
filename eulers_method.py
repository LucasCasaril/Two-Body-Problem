"""
Euler's Method - Range-Kutta 1

Author: Lucas Casaril
"""

import numpy as np
from motion import dydt

def eulers_integration(dydt, t0, tf, y0, G, m1, m2, h):

    lenght = int(tf*(1/h)) 

    # Initial Conditions
    t = t0 + h
    y = y0

    y_result = np.zeros((lenght,12))  # The number of lines is how many iteration the program is going to run. And the number of collumns is 12.
    y_result[0:1, :] = y0 # The fisrt row is always the initial condition

    count = 1 # Access the row number in the Results Matrix - the first row is the initial condition, so we start at the second row

    y_i = y # Initial Condition for the Euler's Method

    time = np.zeros(lenght)

    while t < tf:
        
        # Calling the Differential Equation of the Problem
        f = h * dydt(t,y_i, G, m1, m2)

        t = t + h

        y_inner = y_i + f

        for j in range(12):   # Row starts at 0
            y_result[count, j] = y_inner[j]

        y_i = y_inner # For the next step, we can update Euler's Method

        time[count] += count
        count = count + 1

    return y_result