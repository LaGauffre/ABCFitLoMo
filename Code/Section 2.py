# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     comment_magics: false
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from preamble import *

%config InlineBackend.figure_format = 'retina'
%load_ext lab_black

# +
# dill.load_session("Geometric_Exponential.pkl")

# +
import sys

print("ABC version:", abc.__version__)
print("Python version:", sys.version)
print("Numpy version:", np.__version__)
print("PyMC3 version:", pm.__version__)
print("Arviz version:", arviz.__version__)

tic()
# -

smcArgs = {"numProcs": 40, "verbose": True}

# +
# Create a pseudorandom number generator
rg = default_rng(1234)

# Parameters of the true model
freq = "geometric"
sev = "exponential"
thetaTrue = (0.8, 5)

# Setting the time horizon
T = 100

# Simulating the claim data
freqs, sevs = abc.simulate_claim_data(rg, T, freq, sev, thetaTrue)

# Simulating the observed data
psi = abc.Psi("sum")
xData = abc.compute_psi(freqs, sevs, psi)
# -

params = ("p", "δ")
prior = abc.IndependentUniformPrior([(0, 1), (0, 100)], params)
model = abc.Model(freq, sev, psi, prior)

numIters = 10
popSize = 1000
%time fit = abc.smc(numIters, popSize, xData, model, **smcArgs)

# +
fig, axs = plt.subplots(1, len(params), tight_layout=True)

for l, param in enumerate(params):
    pLims = [prior.marginals[l].isf(1), prior.marginals[l].isf(0)]

    abc.weighted_distplot(fit.samples[:, l], fit.weights, ax=axs[l], hist=False)
    axs[l].legend([], frameon=False)
    axs[l].set_title("$" + param + "$")
    axs[l].set_yticks([])
    axs[l].axvline(thetaTrue[l], color="k", linestyle="dashed", alpha=0.8)
sns.despine(left=True)
# -
%config InlineBackend.figure_format = 'retina'
%load_ext nb_black
numItersData = 15

# ## Trying to recover just the claim sizes conditioned on the observed frequency counts

priorData = abc.IndependentUniformPrior([(0, 100)], ("p", "δ"))
modelData = abc.Model(freqs, sev, psi, priorData)
%time fitData = abc.smc(numItersData, popSize, xData, modelData, **smcArgs)

abc.weighted_distplot(fit.samples[:, 1], fit.weights)
abc.weighted_distplot(fitData.samples[:, 0], fitData.weights)
plt.axvline(thetaTrue[1], c="k", ls="--")
plt.yticks([])
sns.despine(left=True)
plt.legend(["Observing sums", "Observing freqs & sums"], frameon=False)

# ## Using summary statistics

# +
from numba import njit


@njit(nogil=True)
def sumstats(x):
    return np.sum(x)


@njit(nogil=True)
def l1_dist(ssData, ssFake):
    return np.abs(ssData - ssFake)


# +
%%time

fitSS = abc.smc(
    numIters,
    popSize,
    xData,
    model,
    sumstats=sumstats,
    distance=l1_dist,
    **smcArgs
)

# +
basic_model = pm.Model()

with basic_model:
    # Priors for unknown model parameters
    p = pm.Uniform("p", 0, 1)
    δ = pm.Uniform("δ", 0, 100)

    # Log probability of the compound-sum variable
    def logp(t0, sumData):
        return T * np.log(1 - p) + (T - t0) * np.log(p / δ) - (1 - p) / δ * sumData

    exp_surv = pm.DensityDist(
        "X", logp, observed={"t0": np.sum(xData == 0), "sumData": np.sum(xData)}
    )

    %time trace = pm.sample(1000, tune=500, chains = 1, random_seed = 1, return_inferencedata=True)
# -

# ## MLE

# +
t0 = np.sum(xData == 0)
sumData = np.sum(xData)


def logp(θ):
    p, δ = θ
    return T * np.log(1 - p) + (T - t0) * np.log(p / δ) - (1 - p) / δ * sumData


from scipy.optimize import minimize


def costFn(θ):
    return -logp(θ)


bnds = ((0, 1), (0, None))
x0 = np.array((0.5, 6.0))

minRes = minimize(costFn, x0, bounds=bnds)
print("Optimiser reports a success?", minRes.success)

print(minRes.x, thetaTrue)

pPartialMLE = minRes.x[0]
δPartialMLE = minRes.x[1]
# -

# ## MLE using full information

pMLE = 1 - 1 / (freqs.mean() + 1)
δMLE = sevs.mean()

# ## Plot all together

# +
fig, axs = plt.subplots(1, len(params), tight_layout=True)
nMCMC = len(trace.posterior["p"][0])

for l, param in enumerate(params):
    pLims = [prior.marginals[l].isf(1), prior.marginals[l].isf(0)]

    abc.weighted_distplot(fit.samples[:, l], fit.weights, ax=axs[l], hist=False)
    abc.weighted_distplot(fitSS.samples[:, l], fitSS.weights, ax=axs[l], hist=False)
    abc.weighted_distplot(
        trace.posterior[param][0], np.ones(nMCMC) / nMCMC, ax=axs[l], hist=False
    )
    axs[l].legend([], frameon=False)
    axs[l].set_title("$" + param + "$")
    axs[l].set_yticks([])
    axs[l].axvline(thetaTrue[l], color="k", linestyle="dashed", alpha=0.8)

sns.despine(left=True)
# save_cropped("../Figures/geometric-exponential-posterior.pdf")

# +
fig, axs = plt.subplots(1, len(params), tight_layout=True)

for l, param in enumerate(params):
    pLims = [prior.marginals[l].isf(1), prior.marginals[l].isf(0)]

    abc.weighted_distplot(fit.samples[:, l], fit.weights, ax=axs[l], hist=False)
    abc.weighted_distplot(fitSS.samples[:, l], fitSS.weights, ax=axs[l], hist=False)
    abc.weighted_distplot(
        trace.posterior[param][0], np.ones(nMCMC) / nMCMC, ax=axs[l], hist=False
    )
    axs[l].legend([], frameon=False)
    # axs[l].set_title("$" + param + "$")
    axs[l].set_yticks([])
    axs[l].axvline(thetaTrue[l], color="k", linestyle="dashed", alpha=0.8)

draw_prior(prior, axs)
sns.despine(left=True)
save_cropped("../Figures/geometric-exponential-posterior-with-prior.pdf")
# -

elapsed = toc()
print(f"Notebook time = {elapsed:.0f} secs = {elapsed/60:.2f} mins")

# + tags=[]
dill.dump_session("Geometric_Exponential.pkl")
