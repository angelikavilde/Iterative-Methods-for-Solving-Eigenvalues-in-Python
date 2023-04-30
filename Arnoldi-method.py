import numpy as np

def arnoldi(A,g,n):

    #Initialize variables
    m = A.shape[0]
    Q = np.zeros((m,n+1))
    H = np.zeros((n+1,n))
    x = np.zeros(g.shape)

    #Set first column of Q to the normalized input vector g
    Q[:,0] = g /np.linalg.norm(g,2)

    for i in range(1,n+1):
        x = np.dot(Q[:,i-1],A)

        #Compute projection coefficients for previous vectors in Q
        for j in range(0,i):
            H[j,i-1] = np.dot(Q[:,j],x)
            x -= H[j,i-1] * Q[:,j]

        #Compute norm of resulting vector and use to normalize it for the next column in Q
        H[i,i-1] = np.linalg.norm(x,2)
        Q[:,i] = x /np.linalg.norm(x,2)

    return Q, H
