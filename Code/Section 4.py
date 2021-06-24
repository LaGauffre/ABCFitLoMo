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

# # Simulation Study

# +
from preamble import *

%config InlineBackend.figure_format = 'retina'
%load_ext lab_black

# +
# dill.load_session("Sim_Lognormal_Gamma.pkl")

# +
import sys

print("ABC version:", abc.__version__)
print("Python version:", sys.version)
print("Numpy version:", np.__version__)
print("PyMC3 version:", pm.__version__)
print("Arviz version:", arviz.__version__)

tic()

# +
FAST = False

# Processor information and SMC calibration parameters
if not FAST:
    numItersData = 20
    popSize = 1000
    popSizeModels = 1000
    epsMin = 0
    timeout = 1000
else:
    numItersData = 8
    popSize = 500
    popSizeModels = 1000
    epsMin = 1
    timeout = 30

smcArgs = {"timeout": timeout, "epsMin": epsMin, "verbose": True}
smcArgs["numProcs"] = 40
# -

# ## Model selection
#
# We generate individual claim sizes from three models 
#
# - $\text{Weib}\left(k = 1/2,\beta =  e^{1/2}/ \Gamma(3/2)\right)$
# - $\text{Gamma}(r = e^{1/2},m = 1)$
# - $\text{LogNormal}(\mu = 0,\sigma = 1)$
#
# Samples of sizes $25, 50$ are considered

# +
from math import gamma

np.exp(1 / 2) / gamma(3 / 2)

# +
rg = default_rng(123)

sample_sizes = [25, 50, 75, 100, 150, 200]
T = sample_sizes[-1]

claim_data = pd.DataFrame(
    {
        "lognormal": abc.simulate_claim_sizes(rg, T, "lognormal", (0, 1)),
        "gamma": abc.simulate_claim_sizes(rg, T, "gamma", (np.exp(1 / 2), 1)),
        "weibull": abc.simulate_claim_sizes(
            rg, T, "weibull", (1 / 2, np.exp(1 / 2) / gamma(3 / 2))
        ),
    }
)
# -

# ## ABC posterior for choosing between lognormal, Weibull and gamma to model the claim sizes
#
# ### Lognormal data

# +
models_data = ["lognormal"]
models_fitted = ["gamma", "lognormal", "weibull"]

priorG = abc.IndependentUniformPrior([(0, 5), (0, 100)], ("r", "m"))
modelG = abc.Model(sev="gamma", prior=priorG)

priorL = abc.IndependentUniformPrior([(-20, 20), (0, 5)], ("μ", "σ"))
modelL = abc.Model(sev="lognormal", prior=priorL)

priorW = abc.IndependentUniformPrior([(1e-1, 5), (0, 100)], ("k", "δ"))
modelW = abc.Model(sev="weibull", prior=priorW)

models = [modelG, modelL, modelW]

# +
model_proba_abc = pd.DataFrame(
    {"model_data": [], "model_fit": [], "ss": [], "model_probability_ABC": []}
)

# model_data = "lognormal"
for model_data in models_data:
    sevs = claim_data[model_data]
    for ss in sample_sizes:
        uData = sevs[:ss]
        %time fit = abc.smc(numItersData, popSizeModels, uData, models, **smcArgs)
        for k in range(len(models)):
            weights = fit.weights[fit.models == k]
            res_mp = pd.DataFrame(
                {
                    "model_data": pd.Series(model_data),
                    "model_fit": pd.Series([models_fitted[k]]),
                    "ss": np.array([ss]),
                    "model_probability_ABC": pd.Series(
                        np.sum(fit.weights[fit.models == k])
                    ),
                }
            )
            model_proba_abc = pd.concat([model_proba_abc, res_mp])
            model_proba_abc
# -

# ## True posterior samples
#
# We run a Bayesian analysis on the individual claim data and compuet the model probabilities when a Weibull or a gamma distribution is assumed. The prior distribution on the parameters are taken as independent uniform distribution (as in the ABC approach). 
#
# ### Fitting a Weibull and a gamma model to the individual loss data

# +
Bayesian_Summary = pd.DataFrame(
    {
        "model_data": [],
        "model_fit": [],
        "ss": [],
        "param_1": [],
        "param_2": [],
        "marginal_log_likelihood": [],
    }
)

for model_data in models_data:
    sevs = claim_data[model_data]
    for model_fitted in models_fitted:

        for ss in sample_sizes:
            uData = sevs[:ss]
            print(
                f"Fitting a {model_fitted} model to {len(uData)} data points generated from a {model_data} model"
            )

            if model_fitted == "gamma":
                with pm.Model() as model_sev:
                    r = pm.Uniform("param_1", lower=0, upper=5)
                    m = pm.Uniform("param_2", lower=0, upper=100)
                    U = pm.Gamma("U", alpha=r, beta=1 / m, observed=uData)
                    %time trace = pm.sample_smc(popSize, random_seed=1, chains=1)

            elif model_fitted == "lognormal":
                with pm.Model() as model_sev:
                    μ = pm.Uniform("param_1", lower=-20, upper=20)
                    σ = pm.Uniform("param_2", lower=0, upper=5)
                    U = pm.Lognormal("U", mu=μ, sigma=σ, observed=uData)
                    %time trace = pm.sample_smc(popSize, random_seed=1, chains=1)

            elif model_fitted == "weibull":
                with pm.Model() as model_sev:
                    k = pm.Uniform("param_1", lower=1e-1, upper=5)
                    δ = pm.Uniform("param_2", lower=0, upper=100)
                    U = pm.Weibull("U", alpha=k, beta=δ, observed=uData)
                    %time trace = pm.sample_smc(popSize, random_seed=1, chains=1)

            ll = trace.report.log_marginal_likelihood[0]

            res = pd.DataFrame(
                {
                    "model_data": [model_data],
                    "model_fit": [model_fitted],
                    "ss": [ss],
                    "param_1": [trace["param_1"].mean()],
                    "param_2": [trace["param_2"].mean()],
                    "marginal_log_likelihood": [ll],
                }
            )
            Bayesian_Summary = pd.concat([Bayesian_Summary, res])


# -

Bayesian_Summary

# +
max_marginal_log_likelihood = (
    Bayesian_Summary[["model_data", "ss", "marginal_log_likelihood"]]
    .groupby(["model_data", "ss"])
    .max()
)
max_marginal_log_likelihood.reset_index(level=["model_data", "ss"], inplace=True)
max_marginal_log_likelihood.rename(
    columns={"marginal_log_likelihood": "max_marginal_log_likelihood"}
)

Bayesian_Summary_1 = pd.merge(
    Bayesian_Summary, max_marginal_log_likelihood, how="left", on=["model_data", "ss"]
)
Bayesian_Summary_1

Bayesian_Summary_1["BF"] = np.exp(
    Bayesian_Summary_1.marginal_log_likelihood_x
    - Bayesian_Summary_1.marginal_log_likelihood_y
)

Bayesian_Summary_1
sum_BF = (
    Bayesian_Summary_1[["ss", "model_data", "BF"]].groupby(["ss", "model_data"]).sum()
)
sum_BF.reset_index(level=["model_data", "ss"], inplace=True)

Bayesian_Summary_2 = pd.merge(
    Bayesian_Summary_1, sum_BF, how="left", on=["model_data", "ss"]
)
Bayesian_Summary_2
Bayesian_Summary_2["model_probability"] = (
    Bayesian_Summary_2.BF_x / Bayesian_Summary_2.BF_y
)
Bayesian_Summary_2

# +
# # Frequency-Loss Model
# α, p, k, β = 4, 2 / 3, 1 / 3, 1
# rg = default_rng(123)
# uData_10000 = abc.simulate_claim_sizes(rg, 10000, sev, θ_sev)
# r_mle, m_mle, BIC = infer_gamma(uData_10000, [1, 1])

# θ_plot = [[α, p, k, β], [α, p, np.NaN, np.NaN]]
# θ_mle = [[np.NaN, np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, r_mle, m_mle]]
# -

model_proba = pd.merge(
    Bayesian_Summary_2[["model_data", "model_fit", "ss", "model_probability"]],
    model_proba_abc,
    how="left",
    on=["model_data", "model_fit", "ss"],
).round(2)
model_proba


print(
    pd.pivot_table(
        model_proba,
        values=["model_probability", "model_probability_ABC"],
        index=["ss"],
        columns=["model_fit"],
        aggfunc={"model_probability": np.mean, "model_probability_ABC": np.mean},
    ).to_latex()
)

elapsed = toc()
print(f"Notebook time = {elapsed:.0f} secs = {elapsed/60:.2f} mins")

dill.dump_session("Sim_Lognormal_Gamma.pkl")
