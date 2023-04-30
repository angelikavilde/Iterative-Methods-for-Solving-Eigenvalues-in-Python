import numpy as np

def lanczos_iteration(A, g, k):

    m = A.shape[0]
    T = np.zeros((k, k))
    Q = np.zeros((m, k+1))

    #Normalize the initial random vector g
    q = g / np.linalg.norm(g)

    #Setting normalized vector g as the first column of Q
    Q[:, 0] = q

    #Lanczos iteration
    for i in range(k):
        if i > 0:
            #Calculating next orthonormal vector q
            q = r / b
            Q[:, i] = q

        #Calculating the residual vector r
        r = np.dot(A, q)

        #Subtracting projections from residual vector
        if i > 0:
            r -= (b * Q[:, i-1])
            l = np.dot(q, r) #l is alpha to not confuse with matrix A
            r -= (l * q)

        #Calculating beta
        b = np.linalg.norm(r)

        #If the norm of the residual vector is zero, terminate the loop
        if b == 0:
            return Q[:, :i+1], T[:i, :i]

        #Update matrix T
        T[i, i] = l
        if i+1 < k:
            T[i, i+1] = b
            T[i+1, i] = b

    return Q[:, :-1], T
