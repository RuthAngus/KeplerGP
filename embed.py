# Function for the embedding matrix from a set 
# of p lagged versions of the digital signal x[n], 
# where n = N:M
# The matrix is of the form:
#   x[1]      x[2]       x[3]     ...   x[M]
#   x[2]      x[3]       x[4]     ...  x[M-1]
# ..........................................
# x[N-(M-1)] x[N-(M-2)] x[N-(M-3)] ...  x[N]


import numpy as np
import load_data

def embed(x, M, lag):

    # time, flux = load_data.load('012317678', quarter = 3) 
    # x = list(flux[:100])
    # M = 20
    # lag = 1

    x = list(x)

    ematrix = []
    # Create embedding matrix
    for i in range(M):
        row = x[i:]
        for j in range(i):
            row.append(np.mean(x))
        ematrix.append(row)

    ematrix = np.matrix(ematrix)

    return ematrix


#-------------------------------------------------------------------------------------

    # x = np.array(x)
    # N = len(x)

    # hindex = range(0, 20)
    # vindex = range(0, N-(M-1)*lag)

    # Nv = len(vindex)

    # # copies x as a vector into M columns
    # U = np.zeros((M, len(x))) # Columns, rows
    # for i in range(M):
    #     for j in range(len(x)):
    #         U[i][j] = x[j]

    # # replicates the vector 'hindex' on Nv rows
    # a = np.zeros((len(hindex), Nv))
    # for i in range(len(hindex)):
    #     for j in range(Nv):
    #         a[i][j] = hindex[i]

    # # replicates the column vector 'vindex' on M columns
    # b = np.zeros((M, len(vindex))) # Columns, rows
    # for i in range(M):
    #     for j in range(len(vindex)):
    #         b[i][j] = vindex[j]

    # c = a+b

    # # print c
    # # print np.shape(c) # column row

    # for c_row in range(np.shape(c)[1]):
    #     for c_column in range(np.shape(c)[0]):
    #         # print c[c_row][c_column]
    #         print 'yes', U[c[c_row][c_column]]

    # print np.shape(c)
    # print np.shape(U)

    # print U
