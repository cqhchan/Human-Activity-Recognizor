import numpy as np
from statsmodels import regression
from statsmodels import robust
from scipy import stats
from scipy.integrate import simps

# require time domain values i.e. 't' prefix on filename
def mean(array):
    return np.mean(array)

def totalabs(arrayx, arrayy,arrayz):



    total = np.sqrt(pow(arrayx,2) +pow(arrayy,2)+pow(arrayz,2) )

    return np.sum(total)

def std(array):
    return np.std(array)


def mad(array):
    return robust.mad([array], axis=1)


def max(array):
    return np.max(array)


def min(array):
    return np.min(array)


def areatrapz(array):
    return np.trapz(array)

def areasimps(array):
    return simps(array)

def sma(array_x, array_y, array_z):
    sum = 0
    for i in range(array_x.shape[0]):
        sum += (np.abs(array_x[i]) + np.abs(array_y[i]) + np.abs(array_z[i]))
    return sum/array_x.shape[0]


def energy(array):
    sum = 0
    for i in array:
        sum += np.square(i)
    return sum


def iqr(array):
    return stats.iqr(array)


def entropy(array):
    return stats.entropy(array)


def arCoeff(array):
    sigma, rho = regression.yule_walker(array, order=4)
    return sigma, rho


def correlation(array):
    return np.corrcoef(array)


# requires frequency component i.e 'f' prefix on signal name for the following
def maxInds(farray):
    magnitude = np.abs(farray)
    return np.argmax(magnitude)


def meanFreq(farray):
    return mean(farray)


def skewness(farray):
    return stats.skew(farray)


def kurtosis(farray):
    return stats.kurtosis(farray)


# def bandsEnergy(farray):


def angle(vector1, vector2):
    cosang = np.dot(vector1, vector2)/(np.linalg.norm(vector1)+np.linalg.norm(vector2))
    return np.arccos(cosang)

# if __name__ == '__main__':
#     # open file
#     # run methods on file line by line
