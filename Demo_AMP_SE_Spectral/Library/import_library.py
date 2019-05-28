# Sklearn
from sklearn.utils.extmath import randomized_svd
from sklearn.datasets import make_spd_matrix
# Scipy
from scipy import real, imag
from scipy.special import erf, erfc
from scipy.integrate import quad, dblquad, nquad, quadrature, cumtrapz
from scipy.stats import multivariate_normal
import scipy.optimize as optimize
from scipy.linalg import sqrtm, det, norm, eigh, eig
from scipy.sparse.linalg import eigs, eigsh, svds
import scipy.interpolate
from scipy.signal import savgol_filter
# Numpy
import numpy as np
from numpy.random import normal, uniform
from numpy.linalg import inv, det, norm, svd
# Maths
from math import pi, exp, sqrt, log, cosh, sinh, tanh, asin, acos, atan
# Warnings, pickle, platform, socket, os, sys, copy, time, copy, itertools, multiprocessing, collections
import warnings
import pickle
import platform
import socket
import os
import time
import copy
from copy import deepcopy
import itertools
from multiprocessing import Pool
from collections import OrderedDict

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)
