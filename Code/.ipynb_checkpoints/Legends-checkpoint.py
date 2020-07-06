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

# %run -i ./preamble.py
# %config InlineBackend.figure_format = 'retina'
# %load_ext nb_black

lw = 4
plt.rcParams["legend.columnspacing"] = 0.5
plt.rcParams["text.usetex"] = True

from matplotlib.lines import Line2D

# +
priorLine = Line2D([np.NaN], [np.NaN], color=priorColor, lw=lw)
trueLine = Line2D([np.NaN], [np.NaN], **trueStyle)
mleLine = Line2D([np.NaN], [np.NaN], **mleStyle)

labelsPost = ["ABC", "ABC SS", "MCMC"]
linesPost = [Line2D([np.NaN], [np.NaN], c=c, lw=lw) for c in colors[: len(labelsPost)]]

labelsSS = ["ABC (50 $x_s$'s)", "ABC (250 $x_s$'s)"]
labelsSS500 = ["ABC (50 $x_s$'s)", "ABC (500 $x_s$'s)"]
linesSS = [Line2D([np.NaN], [np.NaN], c=c, lw=lw) for c in colors[: len(labelsSS)]]

labelsPrior = ["Prior 1", "Prior 2"]
linesPrior = [Line2D([np.NaN], [np.NaN], c=c, lw=lw) for c in (colors[0], colors[2])]

labelsModel = ["Gamma", "Weibull", "Lognormal"]
linesModel = [
    Line2D([np.NaN], [np.NaN], c=c, lw=lw) for c in colors[: len(labelsModel)]
]


# -

def save_legend(labels, lines, filename, **args):
    fig, ax = plt.subplots(figsize=(3, 0.1))
    ax.legend(lines, labels, ncol=len(labels), frameon=False, **args)
    plt.axis("off")
    save_cropped("../Figures/" + filename)
    plt.show()


save_legend(
    ["Prior"] + labelsPost + ["True"],
    [priorLine] + linesPost + [trueLine],
    "legend-abc-ss-mcmc-prior.pdf",
)

save_legend(
    labelsSS + ["True"], linesSS + [trueLine], "legend-sample-sizes.pdf",
)

save_legend(
    ["Prior"] + labelsSS + ["True"],
    [priorLine] + linesSS + [trueLine],
    "legend-sample-sizes-prior.pdf",
)

# +
# save_legend(
#     ["Prior"] + labelsSS + ["True", "MLE"],
#     [priorLine] + linesSS + [trueLine, mleLine],
#     "legend-sample-sizes-prior-mle.pdf",
# )

# +
save_legend(
    ["Prior"] + labelsSS, [priorLine] + linesSS, "legend-sample-sizes-prior-mle-1.pdf",
)

save_legend(
    ["True", "MLE"], [trueLine, mleLine], "legend-sample-sizes-prior-mle-2.pdf",
)
# -

save_legend(
    labelsSS500 + ["MLE"], linesSS + [mleLine], "legend-sample-sizes-mle.pdf",
)

# +
labels = [
    "ABC (50 $x_i$'s)",
    "ABC (50 $x_i$'s \& $n_i$'s)",
    "ABC (250 $x_i$'s)",
    "ABC (250 $x_i$'s \& $n_i$'s)",
    "True",
]

lines = [
    Line2D([np.NaN], [np.NaN], color=colors[0], lw=lw, alpha=0.6),
    Line2D([np.NaN], [np.NaN], color=colors[0], lw=lw),
    Line2D([np.NaN], [np.NaN], color=colors[1], lw=lw, alpha=0.6),
    Line2D([np.NaN], [np.NaN], color=colors[1], lw=lw),
    trueLine
]

fig, ax = plt.subplots(figsize=(0.5, 0.25))
ax.legend(lines, labels, ncol=3, frameon=False)
plt.axis("off")
save_cropped("../Figures/legend-sample-sizes-true-both.pdf")
# +
labels = [
    "ABC (50 $x_i$'s)",
    "ABC (50 $x_i$'s \& $n_i$'s)",
    "ABC (500 $x_i$'s)",
    "ABC (500 $x_i$'s \& $n_i$'s)",
    "MLE",
]

lines = [
    Line2D([np.NaN], [np.NaN], color=colors[0], lw=lw, alpha=0.6),
    Line2D([np.NaN], [np.NaN], color=colors[0], lw=lw),
    Line2D([np.NaN], [np.NaN], color=colors[1], lw=lw, alpha=0.6),
    Line2D([np.NaN], [np.NaN], color=colors[1], lw=lw),
    mleLine
]

fig, ax = plt.subplots(figsize=(0.5, 0.25))
ax.legend(lines, labels, ncol=3, frameon=False)
plt.axis("off")
save_cropped("../Figures/legend-sample-sizes-mle-both.pdf")
# +
save_legend(
    labelsModel, linesModel, "legend-models.pdf",
)

# fig, ax = plt.subplots(figsize=(0.1, 0.1))
# ax.legend(linesModel, labelsModel, ncol=1, frameon=False)
# plt.axis("off")
# save_cropped("../Figures/legend-models.pdf")
# plt.show()
# -

save_legend(
    labelsPrior, linesPrior, "legend-priors.pdf",
)

save_legend(
    labelsPrior + ["MLE"], linesPrior + [mleLine], "legend-priors-mle.pdf",
)

# +
# labels = [
#     "Prior 1 ($x$'s)",
#     "Prior 1 ($x$'s, $n$'s)",
#     "Prior 2 ($x$'s)",
#     "Prior 2 ($x$'s, $n$'s)",
#     "MLE",
# ]

# lines = [
#     Line2D([np.NaN], [np.NaN], color=colors[0], lw=lw, alpha=0.5),
#     Line2D([np.NaN], [np.NaN], color=colors[0], lw=lw),
#     Line2D([np.NaN], [np.NaN], color=colors[2], lw=lw, alpha=0.5),
#     Line2D([np.NaN], [np.NaN], color=colors[2], lw=lw),
# ] + [mleLine]

# save_legend(
#     labels, lines, "legend-priors-agg-obs.pdf",
# )
# +
# labels = [
#     "Prior 1 ($x$'s)",
#     "Prior 1 ($x$'s, $n$'s)",
#     "Prior 2 ($x$'s)",
#     "Prior 2 ($x$'s, $n$'s)",
# ]

# lines = [
#     Line2D([np.NaN], [np.NaN], color=colors[0], lw=lw, alpha=0.5),
#     Line2D([np.NaN], [np.NaN], color=colors[0], lw=lw),
#     Line2D([np.NaN], [np.NaN], color=colors[2], lw=lw, alpha=0.5),
#     Line2D([np.NaN], [np.NaN], color=colors[2], lw=lw),
# ]

# fig, ax = plt.subplots(figsize=(0.5, 0.25))
# ax.legend(lines, labels, ncol=2, frameon=False)
# plt.axis("off")
# save_cropped("../Figures/legend-priors-agg-obs-1.pdf")

# save_legend(
#     ["MLE"],
#     [mleLine],
#     "legend-priors-agg-obs-2.pdf",
# )
# +
labels = [
    "Prior 1 ($x_i$'s)",
    "Prior 1 ($x_i$'s \& $n_i$'s)",
    "Prior 2 ($x_i$'s)",
    "Prior 2 ($x_i$'s \& $n_i$'s)",
    "MLE",
]

lines = [
    Line2D([np.NaN], [np.NaN], color=colors[0], lw=lw, alpha=0.5),
    Line2D([np.NaN], [np.NaN], color=colors[0], lw=lw),
    Line2D([np.NaN], [np.NaN], color=colors[2], lw=lw, alpha=0.5),
    Line2D([np.NaN], [np.NaN], color=colors[2], lw=lw),
    mleLine
]

fig, ax = plt.subplots(figsize=(0.5, 0.25))
ax.legend(lines, labels, ncol=3, frameon=False)
plt.axis("off")
save_cropped("../Figures/legend-priors-agg-obs.pdf")
