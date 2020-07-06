# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:55:26 2020

@author: Pat and Pierre-O
"""
from scipy.optimize import minimize
import numpy as np

from scipy.special import gamma
from math import pi

import pandas as pd

# this functrion returns the mle estimator and BIC of the Weibull model
def infer_weib(sevs, θ0):
    # sevs is a vector of severities
    # θ0 is a first guess of the parameter values    
    n = len(sevs) 
    def logp(θ):
        k, δ = θ
        return n * np.log(k) - n * k * np.log(δ) + (k-1) * sum(np.log(sevs)) - sum((sevs / δ)**k)
    
    costFn = lambda θ: -logp(θ)
    bnds = ((0, None), (0, None))
    
    minRes = minimize(costFn, θ0, bounds=bnds) #, jac = jac, hess=hess)
    
    BIC = 2 * np.log(n) - 2 * logp(minRes.x) 
    return minRes.x[0], minRes.x[1], BIC


# this function returns the mle estimator and BIC of the gamma model
def infer_gamma(sevs, θ0):
    # sevs is a vector of severities
    # θ0 is a first guess of the parameter values    
    n = len(sevs)
    ΣSevs = sum(sevs)
    ΣlogSevs = sum(np.log(sevs))
    def logp(θ):
        k, δ = θ
        return - n * np.log(gamma(k)) - n * k * np.log(δ) + (k - 1) * ΣlogSevs - ΣSevs / δ 
     
    costFn = lambda θ: -logp(θ)
    bnds = ((0, None), (0, None))
    
    minRes = minimize(costFn, θ0, bounds=bnds)
    
    BIC = 2 * np.log(n) - 2 * logp(minRes.x) 
    return minRes.x[0], minRes.x[1], BIC


# this function returns the mle estimator and BIC of the lognormal model
def infer_lnorm(sevs):
    n = len(sevs)
    θHat =  np.mean(np.log(sevs)), np.std(np.log(sevs))
    def logp(θ):
        mu, σ = θ
        return - n * np.log(σ * (2 * pi)**(1/2)) - sum(np.log(sevs)) - sum((np.log(sevs) - mu)**2 / 2 / σ**2)
    BIC = 2 * np.log(n) - 2 * logp(θHat)
    
    return θHat[0], θHat[1], BIC

    
        
# This function return the mle estimator along with the BIC    
def infer_loss(sevs, models_sev, θ0 = None):
    # sevs vector op loses
    # sev is a claim sizes disribution to chosen in ("weibull", "lognormal",
    # "gamma")
    # θ0 initial guess for the parameter value 
    df = pd.DataFrame()
    for sev in models_sev:
        if sev == "weibull":
            df['weibull'] = np.array(infer_weib(sevs, θ0))
        elif sev == "gamma":
            df['gamma'] = np.array(infer_gamma(sevs, θ0))
        elif sev=="lognormal":
            df['lognormal'] = np.array(infer_lnorm(sevs))            
        else:
            raise Exception("Unknown severity distribution:", sev)
    res = df.transpose()
    res.columns = ['param1', 'param2', 'BIC']
    res['model'] = res.index.values
    res.reset_index(inplace=True, drop = True)
    if len(models_sev) == 1:
        return res
    else:
        BIC_ref = res.BIC.min()
        res['BF'] = np.exp( (BIC_ref - res.BIC.array) / 2)
        res['model_prob'] = res.BF / res.BF.sum()
        return res
            


#Test
# models_sev = ['gamma', 'lognormal', 'weibull']
# len(models_sev)
# rg = Generator(PCG64(1))
# sev = "weibull"
# θ = (2, 5)
# θ0 = [1 , 1]
# sevs = simulate_claim_sizes(rg, 1500, sev, θ)
# infer_loss(sevs, models_sev, θ0)


