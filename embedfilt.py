 # Filter the vector x by making a phase space embedding of it and then
 # projecting onto some singular vectors.

 # x        Vector to be filtered
 # I        Indices of the singular vectors onto which x is projected.
 #           Default [1].
 # D        Dimension in which to embed.  Default 20.

 # y        Filtered x.
 # s        Singular values from embedding of x; X = u s v'
 # u        Singular vectors from embedding.
 # v        Singular vectors from embedding.


import numpy as np
import embed
import scipy

def embedfilt(x):

    # D = len(x)
    D = 20
    L = len(x)
    d = int((D-1)/2) # finds the middle of the embedding dimension
    w = D - 1 - d
    
    x = list(x)
    xx = x[:d]
    [xx.append(x[i]) for i in range(len(x))]
    [xx.append(x[L - w + 1:L][i]) for i in range(len(x[L - w + 1:L]))]
    print len(xx)

    embedded_matrix = embed.embed(xx, D, 1)
    
    u, s, v = np.linalg.svd(embedded_matrix)
    U = u[0]
    print U, len(U)
    print x, len(x)
    c = U*x

    return s
