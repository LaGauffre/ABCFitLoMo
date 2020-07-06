# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Section 4.2

# %run -i ./preamble.py
# %config InlineBackend.figure_format = 'retina'
# %load_ext nb_black

# +
#dill.load_session("Sim_Poisson_Depexp.pkl")

# +
import sys

print("Python version:", sys.version)
print("Numpy version:", np.__version__)
print("PyMC3 version:", pm.__version__)

tic()

# +
FAST = False

# Processor information and SMC calibration parameters
if not FAST:
    numIters = 7
    numItersData = 10
    popSize = 1000
    epsMin = 0
    timeout = 1000
else:
    numIters = 5
    numItersData = 10
    popSize = 500
    epsMin = 1
    timeout = 30

numProcs = 40
smcArgs = {"numProcs": numProcs, "timeout": timeout, "verbose": True}
# -

# ## Inference of a Poison- Frequency Dependent Exponential model
#
# In this notebook we are are conducting a simulation experiment where the claim frequency are Negative Binomial distributed 
#
# $$
# n_s\underset{\textbf{i.i.d.}}{\sim}\text{Pois}(\lambda = 5),\text{ }s = 1,\ldots, 30
# $$ 
#
# and the individual claim sizes are freqency dependent exponential which means that 
#
# $$
# u_1,\ldots, u_{n_s}|n_s\underset{\textbf{i.i.d.}}{\sim}\text{Exp}(\beta\times e^{\delta n_s}),\text{ }s = 1,\ldots 30,
# $$ 
# where we set $\beta = 2$ and $\delta = 0.2$.
#
# The available data is the total claim sizes, we have 
#
# $$
# x_s = \sum_{k = 1}^{n_s}u_k,\text{ }s = 1,\ldots, t.
# $$
#
# Our aim is to see if ABC is able to fit this model featuring dependency between claim counts and claim frequency. 

# +
rg = Generator(PCG64(123))

sample_sizes = [50, 250]
T = sample_sizes[-1]
t = np.arange(1, T + 1, 1)

# Frequency-Loss Model
λ, β, δ = 4, 2, 0.2
θ_True = λ, β, δ
θ_sev = β, δ
θ_freq = λ
sev = "frequency dependent exponential"
freq = "poisson"

# Aggregation process
psi = abcre.Psi("sum")

freqs, sevs = abcre.simulate_claim_data(rg, T, freq, sev, θ_True)
df_full = pd.DataFrame(
    {
        "time_period": np.concatenate([np.repeat(s, freqs[s - 1]) for s in t]),
        "claim_size": sevs,
        "N": np.concatenate([np.repeat(freqs[s - 1], freqs[s - 1]) for s in t]),
    }
)

xData = abcre.compute_psi(freqs, sevs, psi)

df_agg = pd.DataFrame({"time_period": t, "N": freqs, "X": xData})
# -

[np.sum(xData[:ss] > 0) for ss in sample_sizes]

# ## True posterior samples
#
# We run a Bayesian analysis on the individual claim data and frequency data so as to infer the parameters of the Poisson and dependent exponential distribution. The prior distribution on the parameters are taken as independent uniform distribution (as in the ABC approach). 
#
# ### Fitting a dependent exponential model to the individual loss data

# +
dfsev = pd.DataFrame({"ss": [], "β": [], "δ": []})

for ss in sample_sizes:

    fullData = df_full[df_full.time_period <= ss]
    print("The number of individual claim sizes is ", len(fullData))
    with pm.Model() as model:

        # Weakly informative priors,
        β = pm.Uniform("β", lower=0, upper=10)
        δ = pm.Uniform("δ", lower=-1, upper=1)

        # Poisson likelihood
        u = pm.Exponential(
            "u",
            lam=np.exp(-δ * fullData["N"].values) / β,
            observed=fullData["claim_size"].values,
        )
        %time trace = pm.sample_smc(popSize, random_seed=134)
        res = pd.DataFrame(
            {"ss": np.repeat(ss, popSize), "β": trace["β"], "δ": trace["δ"],}
        )

    dfsev = pd.concat([dfsev, res])
# -

# ### Fitting a Poisson model to the claim frequency data

# +
dffreq = pd.DataFrame({"ss": [], "λ": []})

for ss in sample_sizes:
    nData = df_agg.N[df_agg.time_period <= ss]
    with pm.Model() as model_negbin:
        λ = pm.Uniform("λ", lower=0, upper=10)
        N = pm.Poisson("N", mu=λ, observed=nData)

        %time trace = pm.sample_smc(popSize, random_seed=1)
        pm.plot_posterior(trace)

    res = pd.DataFrame({"ss": np.repeat(ss, popSize), "λ": trace["λ"]})
    dffreq = pd.concat([dffreq, res])


# -

# We store all the posterior samples in one single data frame

dftrue = pd.concat([dffreq, dfsev.drop("ss", axis=1)], axis=1)
dftrue["posterior"] = np.repeat("True", len(dftrue))


# ## ABC posterior for the Poisson - frequency dependent exponential model

params = ("λ", "β", "δ")
prior = abcre.IndependentUniformPrior([(0, 10), (0, 20), (-1, 1)], params)
model = abcre.Model("poisson", "frequency dependent exponential", psi, prior)

# +
dfABC = pd.DataFrame({"ss": [], "weights": [], "λ": [], "β": [], "δ": []})

for ss in sample_sizes:
    xDataSS = df_agg.X[df_agg.time_period <= ss].to_numpy()

    %time fit = abcre.smc(numIters, popSize, xDataSS, model, epsMin = epsMin, **smcArgs)

    res = pd.DataFrame(
        {
            "ss": np.repeat(ss, popSize),
            "weights": fit.weights,
            "λ": fit.samples[:, 0],
            "β": fit.samples[:, 1],
            "δ": fit.samples[:, 2],
        }
    )

    dfABC = pd.concat([dfABC, res])


# +
fig, axs = plt.subplots(1, len(params), tight_layout=True)

for l in range(len(params)):
    pLims = [prior.marginals[l].isf(1), prior.marginals[l].isf(0)]
    # axs[l].set_xlim(pLims)

    for k, ss in enumerate(sample_sizes):
        sampleData = dfABC.query("ss == @ss")
        sample = sampleData[params[l]]
        weights = sampleData["weights"]

        dataResampled, xs, ys = abcre.resample_and_kde(sample, weights, clip=pLims)
        axs[l].plot(xs, ys, label="ABC")

    axs[l].axvline(θ_True[l], **trueStyle)
    # axs[l].set_title("$" + params + "$")
    axs[l].set_yticks([])

draw_prior(prior, axs)
sns.despine(left=True)
# save_cropped("../Figures/hist-test2-poisson-depexp.pdf")
# -

# ## ABC posterior for the dependent exponential parameters with the claim frequency

params = ("β", "δ")
prior = abcre.IndependentUniformPrior([(0, 20), (-1, 1)], params)

# +
dfABC_freq = pd.DataFrame({"ss": [], "weights": [], "β": [], "δ": []})

for ss in sample_sizes:
    xDataSS = df_agg.X[df_agg.time_period <= ss].to_numpy()
    nData = np.array(df_agg.N[df_agg.time_period <= ss])

    model = abcre.Model(nData, "frequency dependent exponential", psi, prior)

    %time fit = abcre.smc(numItersData, popSize, xDataSS, model, epsMin = epsMin, **smcArgs)

    res = pd.DataFrame(
        {
            "ss": np.repeat(ss, popSize),
            "weights": fit.weights,
            "β": fit.samples[:, 0],
            "δ": fit.samples[:, 1],
        }
    )

    dfABC_freq = pd.concat([dfABC_freq, res])


# +
fig, axs = plt.subplots(1, len(params), tight_layout=True)

alphas = (0.6, 1)

for l in range(len(params)):
    pLims = [prior.marginals[l].isf(1), prior.marginals[l].isf(0)]

    axs[l].axvline(θ_sev[l], **trueStyle)
    axs[l].set_yticks([])
    for k, ss in enumerate(sample_sizes):

        for j, df in enumerate((dfABC, dfABC_freq)):
            sampleData = df.query("ss == @ss")
            sample = sampleData[params[l]]
            weights = sampleData["weights"]

            dataResampled, xs, ys = abcre.resample_and_kde(sample, weights, clip=pLims)
            axs[l].plot(xs, ys, label="ABC", alpha=alphas[j], c=colors[k])

            # axs[l].set_title("$" + params[l] + "$")

sns.despine(left=True)
# save_cropped("../Figures/hist-test2-poisson-depexp-both.pdf")
# -

elapsed = toc()
print(f"Notebook time = {elapsed:.0f} secs = {elapsed/60:.2f} mins")

dill.dump_session("Sim_Poisson_Depexp.pkl")