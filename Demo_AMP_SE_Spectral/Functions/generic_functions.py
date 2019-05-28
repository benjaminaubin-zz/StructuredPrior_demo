import numpy as np
from math import pi, exp, sqrt
from numpy.linalg import inv, det


def gaussian(x, mean, var):
    """
    returns pdf in x of a multivariate gaussian with mean 'mean' and covariance 'var'

    arguments:
        x: vector where the multivariate gaussian is evaluated
        mean, var: mean and covariance matrix of the gaussian

    example:
        gaussian(0.22, 0, 1)
    """
    x = np.array(x).squeeze()
    mean = np.array(mean).squeeze()
    var = np.array(var)

    try:
        K,  = x.shape
    except:
        K = 1

    if K == 1:
        if var == 0:
            return 1
        else:
            mu = (x-mean)
            return exp(-1/2 * mu**2 / var) / sqrt(2*pi*var)

    else:
        if var.ndim == 0:
            var = np.identity(K) * var
            mean = np.ones(K) * mean
        if det(var) == 0:
            return 1
        else:
            mu = (x-mean)
            try:
                res = exp(- 1/2 * (mu.T).dot(inv(var).dot(mu))) / \
                    sqrt((2*pi)**K * det(var))
            except:
                res = 1
        return res


def relu(x):
    """
    returns REctified Linear Unit of a vector x componentwise

    arguments: 
        x: vector 
    """
    x = np.array(x)
    x[x < 0] = 0
    return x


def sign(x):
    """
    returns Sign of a vector x componentwise

    arguments:
        x: vector
    """
    resul = 1*(x > 0) - 1*(x < 0) + 0*(x == 0)
    resul = resul.astype('float64')
    return resul


def linear(x):
    """
    returns Linear of a vector x componentwise

    arguments:
        x: vector
    """
    return x


def absv(x):
    """
    returns REctified Linear Unit of a vector x componentwise

    arguments: 
        x: vector 
    """
    x = np.array(x)
    x = np.abs(x)
    return x


def sort_lists(X, Y):
    """
    returns two sorted lists
    """

    T = sorted(zip(X, Y), key=lambda x: x[0])
    X = [x for x, _ in T]
    Y = [y for _, y in T]
    return X, Y
