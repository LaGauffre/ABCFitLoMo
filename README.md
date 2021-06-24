# Online accompaniement for the paper "[Approximate Bayesian Computations to fit and compare insurance loss models](https://arxiv.org/abs/2007.03833)"

The Code folder contains the Jupyter notebooks for the examples given in the paper.
Each notebook is named after the corresponding section of the paper.
All the generated plots in our paper are in the Figures folder.

To run the notebooks, we recommend using [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to install the dependencies. 

To install the packages required for our [`approxbayescomp`](https://github.com/Pat-Laub/approxbayescomp) library, use:

```bash
conda install joblib matplotlib numba numpy psutil scipy
pip install fastprogress hilbertcurve
```

Extra packages used in these notebooks are:

```bash
conda install arviz dill pandas pymc3 seaborn
pip install black nb_black pdfCropMargins ttictoc
```

Conda prefers you to run one big install as opposed to multiple installs back-to-back since it can get all the compatible versions in one hit. So, to setup a fresh machine to run these notebooks (and to install Jupyter itself) use:

```
conda install arviz dill joblib matplotlib numba numpy jupyterlab pandas psutil pymc3 scipy seaborn 
pip install black fastprogress nb_black hilbertcurve pdfCropMargins ttictoc
```

PyMC3 has breaking changes to its interface on nearly every version update, so the notebooks have a cell near the top which prints out the version of PyMC3 which we used.
