import numpy as np

def gram_schmidt(v):
    x = np.copy(v)
    x[0] = v[0]/np.linalg.norm(v[0],2)

    for n in range (1,len(v)):
        summation = np.zeros(len(v))
        for k in range (0,n):
            summation += (np.dot(x[k],v[n])/np.linalg.norm(x[k],2))*x[k]
        x[n] = v[n]-summation
        x[n] = x[n]/np.linalg.norm(x[n],2)
    
    return x
