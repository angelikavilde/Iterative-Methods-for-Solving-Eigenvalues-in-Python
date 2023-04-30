import numpy as np

def qr_algorithm(A,k):

    for i in range(k):

        q,r = np.linalg.qr(A)

        A = r@q

    return A
