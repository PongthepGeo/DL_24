#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
#-----------------------------------------------------------------------------------------#
import matplotlib.pyplot as plt
import numpy as np
#-----------------------------------------------------------------------------------------#

np.random.seed(0)
x = 2 * np.random.rand(100, 1) 
y = 4 + 3 * x + np.random.randn(100, 1) 

#-----------------------------------------------------------------------------------------#

learning_rate = 0.05
iterations = 10
m, b = 0, 0 # starting values for m and b

#-----------------------------------------------------------------------------------------#

for i in range(iterations):
    dm, db = U.compute_gradients(m, b, x, y)
    m -= learning_rate * dm
    b -= learning_rate * db
    current_mse = U.compute_mse(m, b, x, y)
    y_pred = m * x + b
    U.basic_gradient_descent(x, y_pred, y, i, m, b, current_mse)

#-----------------------------------------------------------------------------------------#