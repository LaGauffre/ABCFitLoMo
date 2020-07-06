# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 08:44:56 2020

@author: Pat and Pierre-O
"""
# The posterior and model marginal are derived for Poisson, Binomial and negative binomial
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom

# Function that provides the posterior distribution and model marginal for a 
# Poisson distribution with a gamma prior
def infer_pois(freqs, α, β):
    # freq is a vector of count data
    # α is the shape parameter of the gamma prior
    # β is the scale parameter of the gamma prior
    n = len(freqs)
    ΣFreqs = sum(freqs)
    log_margin = sum(np.log(np.arange(1,ΣFreqs + α,1)))\
        - sum([sum(np.log(np.arange(1, freq + 1, 1))) for freq in freqs])\
            - sum(np.log(np.arange(1,α+1,1)))\
                - α * np.log(β) + (ΣFreqs + α) * np.log(β / (n * β +1))  
    posterior_sample = np.random.gamma(shape = ΣFreqs + α , scale = β/ (n * β + 1), size = 400)
    λ_map = (ΣFreqs + α) * β/ (n * β + 1)
    
    def logp(λ):
        return -n * λ + ΣFreqs * np.log(λ)\
            - sum([sum(np.log(np.arange(1, freq + 1, 1))) for freq in freqs])
    
    BIC = - 2 * logp(λ_map) + np.log(n)
    return posterior_sample, log_margin, λ_map, BIC
#test
# rg = Generator(PCG64(5))
# λ = 300
# freqs = rg.poisson(λ, size=100)
# np.mean(freqs)
# α = 100
# β = 100
# infer_pois(freqs, α, β)



def infer_nbinom(freqs, a, b, max_α, size_post_sample):
    # freqs is a vector of count data
    # a and b are the shape parameters of beta prior on p
    # max_α provides a maximu value for the search of the optimal α 
    ΣFreqs = sum(freqs)
    n = len(freqs)
    
    def logp(α, p):
        A = sum([sum(np.log(np.arange(1,α + freq ,1)))- sum(np.log(np.arange(1,α,1)))\
                 - sum(np.log(np.arange(1,freq + 1,1))) for freq in freqs])
        return A + n * α * np.log(p) + ΣFreqs * np.log(1 - p)
    
    map_ps = [ (α, (n * α + a) / (n * α + a + ΣFreqs + b))  for α in np.arange(1, max_α + 1, 1)]
    log_lik = [logp(map_p[0], map_p[1]) for map_p in map_ps] 
    α_star, p_map  = map_ps[log_lik.index(max(log_lik))]
    
    posterior_sample = np.random.beta(a = n * α_star + a , b = ΣFreqs + b, size = size_post_sample)
    
    A = sum([sum(np.log(np.arange(1,α_star + freq ,1)))- sum(np.log(np.arange(1, α_star, 1)))\
                 - sum(np.log(np.arange(1,freq + 1,1))) for freq in freqs])
    B = sum(np.log(np.arange(1, a + b, 1))) - sum(np.log(np.arange(1, a, 1)))\
        - sum(np.log(np.arange(1, b, 1)))
    C = sum(np.log(np.arange(1, n * α_star + a + ΣFreqs + b , 1)))\
        - sum(np.log(np.arange(1, n * α_star, 1)))\
            - sum(np.log(np.arange(1, ΣFreqs + b, 1)))
    log_margin = A + B - C 
    BIC = - 2 * logp(α_star, p_map) + 2 * np.log(n)
    
    return posterior_sample, log_margin,(α_star, p_map), BIC

#Test
# rg = Generator(PCG64(5))
# α = 3 
# p = 0.23
# freqs = rg.negative_binomial(α, p, size=10)
# np.mean(freqs==0), p**α
# max_α = 5
# a = 10
# b = 10
# infer_nbinom(freqs, 30, 30, max_α)
# infer_pois(freqs, 100, 100)


