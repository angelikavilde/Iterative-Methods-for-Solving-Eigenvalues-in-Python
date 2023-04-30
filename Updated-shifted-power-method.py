import numpy as np

def shifted_power_method(A):

    #Creating guess vector g
    g = np.zeros(A.shape[0])
    for i in range (A.shape[0]):
        g[i] = 1

    x = g ; y = g
    r = 30 #number of iterations until termination

    #Power method for dominant eigenvector
    for j in range (r):
        x = np.dot(A,x) ; y = np.dot(A,y)
        x = x/np.linalg.norm(x,2) ; y = y/np.linalg.norm(y,2)
        
        #Forced termination of the loop if values are close by 0.01
        if np.isclose(x, y, rtol= 1e-2, atol= 1e-2):
            break

    #Rayleigh quotient for dominant eigenvalue
    lambda1 = np.dot(np.dot(A,x),x)/np.dot(x,x)

    #Creating a list for eigenvalues of A
    eig = np.zeros(A.shape[0])
    eig[0] = lambda1

    #Creating matrix B
    for k in range (1,A.shape[0]):
        B = A-(eig[k-1]*np.identity(A.shape[0]))
        x = g ; y = g

        #Power method for subsequent eigenvectors
        for z in range (r):
            x = np.dot(B,x) ; y = np.dot(A,y)
            x = x/np.linalg.norm(x,2) ; y = y/np.linalg.norm(y,2)

            #Forced termination of the loop if values are close by 0.01
            if np.isclose(x,y, rtol= 1e-2, atol=1e-2):
                break

        #Rayleigh quotient for subsequent eigenvalues
        lambdaofB = np.dot(np.dot(B,x),x)/np.dot(x,x)

        #Calculating next eigenvalue
        eig[k] = lambdaofB + eig[k-1]

    return eig
