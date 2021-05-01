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

# # Simulation Study

%run -i ./preamble.py
%run -i ./infer_loss_distribution.py
%run -i ./infer_count_distribution.py
%config InlineBackend.figure_format = 'retina'
%load_ext nb_black

# +
#dill.load_session("Sim_Weibull_Gamma.pkl")

# +
import sys

print("Python version:", sys.version)
print("Numpy version:", np.__version__)
print("PyMC3 version:", pm.__version__)
print("Arviz version:", arviz.__version__)

tic()

# +
FAST = False

# Processor information and SMC calibration parameters
if not FAST:
    numIters = 7
    numItersData = 10
    popSize = 1000
    popSizeModels = 1000
    epsMin = 0
    timeout = 1000
else:
    numIters = 4
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
# In this notebook we are are conducting a simulation experiment where the claim frequency are Negative Binomial distributed 
#
# $$
# n_s\underset{\textbf{i.i.d.}}{\sim}\text{Neg-Bin}(\alpha = 4, p = 2/3),\text{ }s = 1,\ldots, 30
# $$ 
#
# and the individual claim sizes are weibull distributed
#
# $$
# u_1,\ldots, u_{n_s}\underset{\textbf{i.i.d.}}{\sim}\text{Weib}(k = 1/2, \beta = 1),\text{ }s = 1,\ldots 30.
# $$ 
#
# The available data is aggregated claim sizes in excess of the priority $c=1$ asociated to aa global stop-loss treaty, we have 
#
# $$
# x_s = \left(\sum_{k = 1}^{n_s}u_k-c\right)_{+},\text{ }s = 1,\ldots, t.
# $$
#
# Our aim is to look into the finite sample performance of our ABC implementation when deciding which model is the most suited.

# +
rg = default_rng(123)

sample_sizes = [50, 250]
T = sample_sizes[-1]
t = np.arange(1, T + 1, 1)

# Frequency-Loss Model
α, p, k, β = 4, 2/3, 1/3, 1
θ_True = α, p, k, β
θ_sev = k, β
θ_freq = α, p
freq = "negative binomial"
sev = "weibull"

# Aggregation process
c = 1
psi = abcre.Psi("GSL", c)

freqs, sevs = abcre.simulate_claim_data(rg, T, freq, sev, θ_True)
df_full = pd.DataFrame(
    {
        "time_period": np.concatenate([np.repeat(s, freqs[s - 1]) for s in t]),
        "claim_size": sevs,
    }
)

xData = abcre.compute_psi(freqs, sevs, psi)

df_agg = pd.DataFrame({"time_period": t, "N": freqs, "X": xData})
# -

[np.sum(xData[:ss] > 0) for ss in sample_sizes]

# ## True posterior samples
#
# We run a Bayesian analysis on the individual claim data and compuet the model probabilities when a Weibull or a gamma distribution is assumed. The prior distribution on the parameters are taken as independent uniform distribution (as in the ABC approach). 
#
# ### Fitting a Weibull and a gamma model to the individual loss data

# +
Bayesian_Summary = pd.DataFrame({"model": [], "ss": [], "k": [], "β": []})
models = ["True weibull", "True gamma"]
for m in models:
    for ss in sample_sizes:

        uData = np.array(df_full.claim_size[df_full.time_period <= ss])
        print("The number of individual claim sizes is ", len(uData))
        if m == "True weibull":
            # We fit a Weibull model using SMC
            with pm.Model() as model_sev:
                k = pm.Uniform("k", lower=1e-1, upper=10)
                β = pm.Uniform("β", lower=0, upper=20)
                U = pm.Weibull("U", alpha=k, beta=β, observed=uData)
                %time trace = pm.sample_smc(popSize, random_seed=1, chains=1) 

        elif m == "True gamma":
            # We fit a gamma model using SMC
            with pm.Model() as model_sev:
                param1 = pm.Uniform("k", lower=0, upper=10)
                param2 = pm.Uniform("β", lower=0, upper=50)
                U = pm.Gamma("U", alpha=param1, beta=1 / param2, observed=uData)
                %time trace = pm.sample_smc(popSize, random_seed=1, chains=1)

        arviz.plot_posterior(trace)

        log_lik = trace.report.log_marginal_likelihood[0]

        res = pd.DataFrame(
            {
                "model": [m],
                "ss": [ss],
                "k": [trace["k"].mean()],
                "β": [trace["β"].mean()],
                "marginal_log_likelihood": [log_lik],
            }
        )
        Bayesian_Summary = pd.concat([Bayesian_Summary, res])

max_marginal_log_likelihood = (
    Bayesian_Summary[["ss", "marginal_log_likelihood"]]
    .groupby("ss")
    .max()
    .marginal_log_likelihood.values
)
max_marginal_log_likelihood = np.concatenate(
    [max_marginal_log_likelihood, max_marginal_log_likelihood]
)
Bayesian_Summary["BF"] = np.exp(Bayesian_Summary.marginal_log_likelihood - max_marginal_log_likelihood)
sum_BF = Bayesian_Summary[["ss", "BF"]].groupby("ss").sum().BF.values
sum_BF = np.concatenate([sum_BF, sum_BF])
Bayesian_Summary["model_probability"] = Bayesian_Summary.BF / sum_BF
Bayesian_Summary

# +
# Frequency-Loss Model
α, p, k, β = 4, 2 / 3, 1 / 3, 1
rg = default_rng(123)
uData_10000 = abcre.simulate_claim_sizes(rg, 10000, sev, θ_sev)
r_mle, m_mle, BIC = infer_gamma(uData_10000, [1, 1])

θ_plot = [[α, p, k, β], [α, p, np.NaN, np.NaN]]
θ_mle = [[np.NaN, np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, r_mle, m_mle]]
# -

# ## ABC posterior for choosing between Weibull and gamma to model the claim sizes

# +
params = (("α", "p", "k", "β"), ("α", "p", "r", "m"))

prior1 = abcre.IndependentUniformPrior([(0, 20), (1e-3, 1), (1e-1, 10), (0, 20)], params[0])
model1 = abcre.Model("negative binomial", "weibull", psi, prior1)

prior2 = abcre.IndependentUniformPrior([(0, 20), (1e-3, 1), (1e-1, 10), (0, 50)], params[1])
model2 = abcre.Model("negative binomial", "gamma", psi, prior2)

models = [model1, model2]
model_names = ["ABC negative binomial - weibull", "ABC negative binomial - gamma"]

# +
model_proba_abc = pd.DataFrame({"model": [], "ss": [], "model_probability": []})
dfabc = pd.DataFrame(
    {"model": [], "ss": [], "weights": [], "α": [], "p": [], "param1": [], "param2": []}
)

for ss in sample_sizes:
    xDataSS = df_agg.X[df_agg.time_period <= ss].values

    %time fit = abcre.smc(numIters, popSizeModels, xDataSS, models, **smcArgs)

    for k in range(len(models)):
        weights = fit.weights[fit.models == k]
        res_mp = pd.DataFrame(
            {
                "model": pd.Series([model_names[k]]),
                "ss": np.array([ss]),
                "model_probability": pd.Series(np.sum(fit.weights[fit.models == k])),
            }
        )

        model_proba_abc = pd.concat([model_proba_abc, res_mp])

        res_post_samples = pd.DataFrame(
            {
                "model": np.repeat(model_names[k], len(weights)),
                "ss": np.repeat(ss, len(weights)),
                "weights": weights / np.sum(weights),
                "α": np.array(fit.samples)[fit.models == k, 0],
                "p": np.array(fit.samples)[fit.models == k, 1],
                "param1": np.array(fit.samples)[fit.models == k, 2],
                "param2": np.array(fit.samples)[fit.models == k, 3],
            }
        )
        dfabc = pd.concat([dfabc, res_post_samples])
# -

for l in range(len(models)):
    fig, axs = plt.subplots(1, len(params[l]), tight_layout=True)
    prior = models[l].prior

    for k in range(len(params[l])):
        pLims = [prior.marginals[k].isf(1), prior.marginals[k].isf(0)]
        axs[k].set_xlim(pLims)

        for i, ss in enumerate(sample_sizes):
            selector = (dfabc.ss == ss) & (dfabc.model == model_names[l])
            sample = np.array(dfabc[["α", "p", "param1", "param2"]])[selector, k]
            weights = dfabc.weights[selector].values
            dataResampled, xs, ys = abcre.resample_and_kde(
                sample, weights / sum(weights), clip=pLims
            )
            axs[k].plot(xs, ys)
            axs[k].axvline(θ_plot[l][k], **trueStyle)
            axs[k].axvline(θ_mle[l][k], **mleStyle)

            axs[k].set_title("$" + params[l][k] + "$")
            axs[k].set_yticks([])
            
    sns.despine(left=True)
    #plt.save_cropped(f"../Figures/hist-negbin-weibull-model-selection-{l}.pdf")

# +
params = (("k", "β"), ("r", "m"))

prior1 = abcre.IndependentUniformPrior([(1e-1, 10), (0, 20)], params[0])
prior2 = abcre.IndependentUniformPrior([(0, 10), (0, 50)], params[1])

model_names = ("ABC with freqs - weibull", "ABC with freqs - gamma")

# +
model_proba_abc_freq = pd.DataFrame({"model": [], "ss": [], "model_probability": []})
dfabc_freq = pd.DataFrame(
    {"model": [], "ss": [], "weights": [], "param1": [], "param2": []}
)

for ss in sample_sizes:
    xDataSS = df_agg.X[df_agg.time_period <= ss].values
    nData = df_agg.N[df_agg.time_period <= ss].values

    model1 = abcre.Model(nData, "weibull", psi, prior1)
    model2 = abcre.Model(nData, "gamma", psi, prior2)
    models = [model1, model2]

    %time fit = abcre.smc(numItersData, popSizeModels, xDataSS, models, **smcArgs)

    for k in range(len(models)):
        weights = fit.weights[fit.models == k]
        res_mp = pd.DataFrame(
            {
                "model": pd.Series([model_names[k]]),
                "ss": np.array([ss]),
                "model_probability": pd.Series(np.sum(fit.weights[fit.models == k])),
            }
        )

        model_proba_abc_freq = pd.concat([model_proba_abc_freq, res_mp])

        res_post_samples = pd.DataFrame(
            {
                "model": np.repeat(model_names[k], len(weights)),
                "ss": np.repeat(ss, len(weights)),
                "weights": weights / np.sum(weights),
                "param1": np.array(fit.samples)[fit.models == k, 0],
                "param2": np.array(fit.samples)[fit.models == k, 1],
            }
        )
        dfabc_freq = pd.concat([dfabc_freq, res_post_samples])
# -


for l in range(len(models)):
    modelName = model_names[l]
    prior = models[l].prior
    fig, axs = plt.subplots(1, len(params[l]), tight_layout=True)

    for k in range(len(params[l])):
        pLims = [prior.marginals[k].isf(1), prior.marginals[k].isf(0)]
        # axs[k].set_xlim(pLims)

        for ss in sample_sizes:
            sampleData = dfabc_freq.query("ss == @ss & model == @modelName")
            selector = (dfabc_freq.ss == ss) & (dfabc_freq.model == model_names[l])
            sample = np.array(dfabc_freq[["param1", "param2"]])[selector, k]
            weights = dfabc_freq.weights[selector].values
            if sampleData.shape[0] > 1:
                dataResampled, xs, ys = abcre.resample_and_kde(
                    sample, weights / sum(weights), clip=pLims
                )
                axs[k].plot(xs, ys)

            axs[k].axvline(θ_plot[l][k + 2], **trueStyle)
            axs[k].axvline(θ_mle[l][k + 2], **mleStyle)

            axs[k].set_title(
               "$" + params[l][k] + f"\\in ({pLims[0]:.0f}, {pLims[1]:.0f})$"
            )
            axs[k].set_yticks([])

    sns.despine(left=True)
    #plt.save_cropped(f"../Figures/hist-freqs-weibull-model-selection-{l}.pdf")
# +
model_proba_df = pd.concat(
    [
        Bayesian_Summary[["model", "ss", "model_probability"]],
        model_proba_abc,
        model_proba_abc_freq,
    ]
)
model_proba_df = model_proba_df[
    np.char.find(model_proba_df.model.tolist(), "weibull") > -1
]

model_proba_df.model = model_proba_df.model.replace(
    {
        "True weibull": "True\n(w/ $N$'s, $U$'s)",
        "ABC negative binomial - weibull": "ABC\n(w/ $X$'s)",
        "ABC with freqs - weibull": "ABC\n(w/ $X$'s, $N$'s)",
    }
)

model_proba_df = model_proba_df.sort_values("model")

fig, ax = plt.subplots(1, 1, tight_layout=True)

g = sns.barplot(
    x="model",
    y="model_probability",
    hue="ss",
    data=model_proba_df,
    #     legend=False,
    ax=ax,
)
plt.legend([], frameon=False)
plt.ylabel("")
plt.xlabel("")
plt.title("")

sns.despine()
#save_cropped("../Figures/barplot-negbin-weibull-model-selection.pdf")
# -
model_proba_df = pd.concat(
    [
        Bayesian_Summary[["model", "ss", "model_probability"]],
        model_proba_abc,
        model_proba_abc_freq,
    ]
)
print(
    pd.pivot_table(
        model_proba_df,
        values="model_probability",
        index=["ss"],
        columns=["model"],
        aggfunc=np.sum,
    ).to_latex()
)


elapsed = toc()
print(f"Notebook time = {elapsed:.0f} secs = {elapsed/60:.2f} mins")

dill.dump_session("Sim_Weibull_Gamma.pkl")
