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
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Show ABC-AR with varying levels of acceptance

%run -i ./preamble.py
%config InlineBackend.figure_format = 'retina'
%load_ext nb_black

# +
import sys

print("Python version:", sys.version)
print("Numpy version:", np.__version__)

tic()
# -

popSize = 1000

# +
# Create a pseudorandom number generator
rg = default_rng(1234)

thetaTrue = (0, 5)
T = 250

xData = rg.normal(thetaTrue[0], thetaTrue[1], T)
xData.mean(), xData.std()

# +
from numba import njit
from numpy.random import SeedSequence, default_rng
from tqdm.notebook import trange


@njit()
def numba_seed(seed):
    rnd.seed(seed)


@njit(nogil=True)
def uniform_sampler(lower, width):
    d = len(lower)
    theta = np.empty(d, np.float64)
    for i in range(d):
        theta[i] = lower[i] + width[i] * rnd.random()
    return theta


def distance(x, y):
    return np.linalg.norm(np.sort(x) - np.sort(y))


def sample(seed, priorLower, priorWidth, eps):
    rg = default_rng(seed)
    numba_seed(seed)
    attempts = 0

    while True:
        attempts += 1
        theta = uniform_sampler(priorLower, priorWidth)
        xFake = rg.normal(theta[0], theta[1], T)
        dist = distance(xData, xFake)

        if dist < eps:
            break

    return theta, attempts


def abc_ar(priorLower, priorWidth, eps, popSize, seed):
    sg = SeedSequence(seed)
    seeds = [s.generate_state(1)[0] for s in sg.spawn(popSize)]
    particles = np.empty((popSize, 2), np.float64)
    attempts = 0
    for p in trange(popSize):
        theta, attempts_p = sample(seeds[p], priorLower, priorWidth, eps)
        particles[p, :] = theta
        attempts += attempts_p

    return particles, attempts


# +
priorLower = np.array([-10, 0])
priorWidth = np.array([20, 10])

seed = 1
eps = 10
%time sample(seed, priorLower, priorWidth, eps)
# -

%time samples, att = abc_ar(priorLower, priorWidth, eps, popSize, seed)
att

# +
fig, axs = plt.subplots(1, 2, tight_layout=True)
prior = abcre.IndependentUniformPrior([(-10, 10), (0, 10)], ("mu", "sigma"))

for eps in [50, 25, 10]:
    %time samples, att = abc_ar(priorLower, priorWidth, eps, popSize, seed)

    weights = np.ones(popSize) / popSize

    pLims = [prior.marginals[0].isf(1), prior.marginals[0].isf(0)]
    _, xs, ys = abcre.resample_and_kde(samples[:, 0], weights, clip=pLims)
    axs[0].plot(xs, ys)

    axs[0].axvline(thetaTrue[0], c="k", ls="--")
    axs[0].set_yticks([])
    sns.despine(left=True)

    # plt.subplot(1, 2, 2)
    weights = np.ones(popSize) / popSize

    pLims = [prior.marginals[1].isf(1), prior.marginals[1].isf(0)]
    _, xs, ys = abcre.resample_and_kde(samples[:, 1], weights, clip=pLims)
    axs[1].plot(xs, ys)

    axs[1].axvline(thetaTrue[1], c="k", ls="--")
    axs[1].set_yticks([])
    sns.despine(left=True)


draw_prior(prior, axs)

save_cropped("../Figures/smc-intro.pdf")
