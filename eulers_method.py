"""
Euler's Method - Range-Kutta 1
"""

import numpy as np
from motion import dydt

def eulers_integration(dydt, t0, tf, y0, G, m1, m2):

    stages = 1 # Number of points within the time interval
    a = 0
    b = 0
    c = 1
    h = 1 ############# E se o passo for um numero quebrado ? - Tamanho da Matrix ################

    # Initial Conditions
    t = t0
    y = y0

    y_result = np.zeros((tf*h,12))  # The number of lines is how many iteration the program is going to run. And the number of collumns is 12.
    y_result[0:1, :] = y0 # The fisrt row is always the initial condition

    count = 1 # Access the row number in the Results Matrix - the first row is the initial condition, so we start at the second row

    y_i = y # Initial Condition for the Euler's Method

    time = np.zeros(tf*h)

    while t < tf:
        
        # Calling the Differential Equation of the Problem
        f = dydt(t,y_i, G, m1, m2)

        t = t + h

        y_inner = y_i + h * f

        for j in range(12):   # Row starts at 0
            y_result[count, j] = y_inner[j]

        y_i = y_inner # For the next step, we can update Euler's Method

        time[count] += count
        count = count + 1

    return y_result