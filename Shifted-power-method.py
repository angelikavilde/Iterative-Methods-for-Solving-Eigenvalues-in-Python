import numpy as np
def shifted_power_method(A,g,k,lambda1):

    eig = np.zeros(A.shape[0])
    eig[0] = lambda1

    for i in range (1,A.shape[0]):

        B = A-(eig[i-1]*np.identity(A.shape[0]))
        x = g

        for j in range (k):
            x = np.dot(B,x)
            x = x/np.linalg.norm(x,2)
        lambdaofB = np.dot(np.dot(B,x),x)/np.dot(x,x)

        eig[i] = lambdaofB + eig[i-1]

    return eig
