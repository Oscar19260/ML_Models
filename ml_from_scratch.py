"""
This is a simple ML algorithm for linear regression made from scratch.
    - Hypothesis function h(x) = m*x + b
    - Cost function J(theta) = 1/2m * sum((h(x) - y)^2)
    - Gradient Descent algorithm
"""

import numpy as np  #numpy is used to work with arrays and make processing faster
import pandas as pd #extract data from a file
import matplotlib.pyplot as plt #graphic the mean square error
from sklearn.model_selection import train_test_split #get a train and test as params


__errors__ = []

def hyp(m_b, x):
    """
        This evaluates a generic linear function h(x) 
        with current parameters.
    """
    acum = 0
    size_mb = len(m_b)
    for i in range(size_mb):
        acum = acum + m_b[i]*x[i]
    return acum

def cost(m_b, x, y):
    """
        This calculate the cost function with MSE
        J(theta) = 1/2m * sum((h(x) - y)^2)
    """
    acum = 0
    size = len(x)
    for i in range(size):
        h = hyp(m_b, x[i])
        # print( "hyp  %f  y %f " % (h,  y[i]))  
        error = h - y[i]
        acum = acum + error**2
    __errors__.append(acum/size)

def gradient_descent(m_b, x, y, alpha):
    """
        This apply Gradent Descent algorithm:
        repeat until convergence{
            theta[j] = theta[j] - alpha*(d/d_theta)*J(theta_0, theta_1) 
        } (for j=0 and j=1)
    """
    size_mb = len(m_b)
    size_x = len(x)
    theta = list(m_b)
    for i in range(size_mb):
        acum = 0
        for j in range(size_x):
            error = hyp(m_b, x[j]) - y[j]
            acum = acum + error*x[j][i]
        theta[i] = theta[i] - alpha*(1/size_x)*acum
    return theta

def min_max_scaler(x):
    """
        This function scale the data to a range between 0 and 1
    """
    return (x - np.min(x))/(np.max(x) - np.min(x))

def add_ones(x):
    """
        This functions add ones to been capable of make the operations
    """
    for i in range(len(x)):
        x[i].append(1)
    return x


if __name__ == "__main__":
    # Test with dataset
    cols = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    df = pd.read_csv('datasets/abalone.data', names=cols)
    df_x = df.iloc[:, 5:8].to_numpy().tolist()
    df_y = df.iloc[:, 4].to_numpy().tolist()
    params = np.zeros(len(df_x[0]) + 1).tolist()

    # split data into test and train params
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, random_state=1)

    a = 0.05  # learning rate
    epochs = 0

    x_train = add_ones(x_train)
    
    # Make the train #############################################################################################
    while True:  # run gradient descent until local minima is reached
        oldparams = list(params)
        params = gradient_descent(params, x_train, y_train, a)	
        cost(params, x_train, y_train)  # only used to show errors, it is not used in calculation
        epochs += 1
        if(oldparams == params or epochs == 2000):   # local minima is found when there is no further improvement
            # print ("samples:")
            # print(x_train)
            print ("final params:")
            print (params)            
            print("error")
            print(__errors__[-1])
            break

    plt.plot(__errors__)

    # Check the test #############################################################################################
    x_test = add_ones(x_test)
    yp = [np.dot(x, params) for x in x_test]

    acum_e = 0
    for i in range(len(y_test)):
        acum_e = acum_e + (y_test[i] - yp[i]) ** 2
    
    print("Test error:")
    print(acum_e)

    plt.show()
