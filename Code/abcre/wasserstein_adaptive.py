import numpy as np


def wass_adap_sumstats(x):
    return (np.sum(x == 0), np.sort(x))


def wass_adap_dist(ssData, ssFake, t=0, p=1.0):
    numZerosData, sortedData = ssData
    numZerosFake, sortedFake = ssFake
    zerosDiff = np.abs(numZerosData - numZerosFake)
    
    # Enforce the zero-matching after a few iterations of SMC
    if t >= 5 and zerosDiff != 0:
        return np.inf

    # For early iterations, just put a penalty on zeros not matching
    n = len(sortedData)
    zerosDiffPenalty = np.exp(t * zerosDiff)
    lpDist = np.linalg.norm(sortedData - sortedFake, p) / n

    return zerosDiffPenalty * lpDist