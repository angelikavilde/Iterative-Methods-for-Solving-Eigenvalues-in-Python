import numpy as np
def power_iteration(A,g,k):

    for i in range (1,k):
        g = np.dot(A,g)
        x = max(g)
        g = g/x

    eig = np.dot(np.dot(A,g),g)/np.dot(g,g)
    return eig
