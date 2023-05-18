import numpy as np

# inhomogeneous point (x,y,z)
# homogeneous point (x,y,z,s)

def inv_PI(P):
    """ Convert from inhomogeneous to homogeneous coordinates"""
    return np.vstack((P, np.ones(len(P[0]))))

def PI(P):
    """ Convert from homogeneous to inhomogeneous coordinates"""
    #if P.shape[1]!= None and P.shape[1] == 1:
     #   return P[:-1] / P[-1]
    return P[:-1, :] / P[-1,:]