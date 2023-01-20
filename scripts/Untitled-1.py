# %%

import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import arviz
import os
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.contrib.examples.util  # patches torchvision
from pyro.contrib.examples.util import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import copy
from IPython.display import clear_output

gp_fig_folder = "../report/figures/gp/"

os.makedirs(os.path.dirname(gp_fig_folder), exist_ok=True)


# %% [markdown]
# # B.1

# %% [markdown]
# ## Data

# %%
def f(x):
    return torch.sin(20 * x) + 2 * torch.cos(14 * x) - 2 * torch.sin(6 * x)
X = torch.tensor([-1, -1/2, 0, 1/2, 1])
y = f(X)
x = torch.tensor([-1/4])
XNew = torch.linspace(-1, 1, steps=200)

# %% [markdown]
# ## Gaussian process definition

# %%
def make_gpr(X,y, kernel = None, prior_dict = None):
    pyro.clear_param_store()
    if kernel is None:
        kernel = gp.kernels.RBF(input_dim=1)
        kernel.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 2.0))
        kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(-1.0, 1.0))
    else:
        assert kernel is not None
        assert isinstance(prior_dict, dict)
        for attr,prior in prior_dict.items():
            setattr(kernel, attr, prior)
    return gp.models.GPRegression(X, y, kernel, noise=torch.tensor(10**(-4)))

# %% [markdown]
# ## MCMC hyper-parameters

# %%
C = 4
W = 100
N = 500
#C = 2
#W = 1
#N = 4

# %% [markdown]
# ## Sampling posterior of GP with MCMC

# %%
def mcmc_sampler(gpr, C, W, N):
    nuts_kernel = pyro.infer.NUTS(gpr.model)
    mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=N,
                        num_chains=C, warmup_steps = W, mp_context = "spawn")
    mcmc.run()
    return mcmc

# %%
gpr = make_gpr(X, y)
mcmc = mcmc_sampler(gpr, C, W, N)

# %% [markdown]
# ## Estimates of mean and variance based on samples

# %%
def predictive(x, gpr, prior_dict = None):
    if prior_dict is None:
        gpr.kernel.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 2.0))
        gpr.kernel.lengthscale = pyro.nn.module.PyroSample(dist.LogNormal(-1.0, 1.0))
    else:
        assert isinstance(prior_dict, dict)
        for attr,prior in prior_dict.items():
            setattr(gpr.kernel, attr, prior)
    loc, cov = gpr(x, noiseless = False, full_cov = True)
    var = cov.diag()
    pyro.sample("loc", dist.Delta(loc))
    pyro.sample("var", dist.Delta(var))
    pyro.sample("f", dist.Normal(loc, var))

# %%
def sample_predict(x, gpr, mcmc, default = True, prior_dict = None):
    posterior_samples=mcmc.get_samples(group_by_chain=True)
    posterior_samples_flat = {k : v.flatten() for (k,v) in posterior_samples.items()}
    posterior_predictor = pyro.infer.Predictive(predictive, posterior_samples = posterior_samples_flat)
    posterior_predictive = posterior_predictor(x,gpr, prior_dict)
    return posterior_predictive, posterior_samples

# %%
posterior_predictive, posterior_samples = sample_predict(XNew, gpr, mcmc)
posterior_locs = posterior_predictive['loc']
posterior_vars = posterior_predictive['var']
print(posterior_locs.shape, posterior_vars.shape)
posterior_lengthscale = posterior_samples['kernel.lengthscale']
posterior_variance = posterior_samples['kernel.variance']

# %% [markdown]
# ## A scatter plot on log-log scale of N = 500 samples from $P(\theta | \mathcal{D})$

# %%
for c in range(C):
    plt.scatter(posterior_lengthscale[c],posterior_variance[c], label = "chain " + str(c))
plt.xlabel(r"$\sigma_l^2$", fontsize = 15)
plt.ylabel(r"$\sigma_v$", fontsize=15)
plt.yscale("log")
plt.xscale("log")
plt.legend()
slim=0.6
plt.tight_layout(pad=-slim, w_pad=-slim, h_pad=-slim)
plt.savefig(gp_fig_folder + "loglogscale.pdf",
            bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Sample quality analysis

# %%
data = arviz.from_pyro(mcmc)
summary = arviz.summary(data, hdi_prob=0.95)
print(summary)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
arviz.plot_posterior(data, hdi_prob=0.95, ax=axs)
slim=0.6
plt.tight_layout(pad=-slim, w_pad=-slim, h_pad=-slim)
plt.savefig(gp_fig_folder + "sample_analysis.pdf",
            bbox_inches="tight")
plt.show()


# %% [markdown]
# ## Plot visualizing $p(f^* | x^*, \mathcal{D})$

# %%
plt.plot(XNew, f(XNew), label = r"$f(x)$")
plt.scatter(X, y, color = 'tab:red', label = r"$\mathcal{D}$")
print(posterior_locs.mean(dim=0).shape)
plt.plot(XNew, posterior_locs.mean(dim=0), color = 'tab:green', label = r"$m(x^*)$")
plt.plot(XNew, posterior_locs.mean(dim=0) + 2*np.sqrt(posterior_vars.mean(dim=0)), color='tab:purple', linestyle='dotted', label=r"$m(x^*) + 2\sqrt{v(x^*)}$")
plt.plot(XNew, posterior_locs.mean(dim=0) -2*np.sqrt(posterior_vars.mean(dim=0)), color='tab:purple', linestyle='dashed', label=r"$m(x^*) -2\sqrt{v(x^*)}$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.legend()
slim=0.6
plt.tight_layout(pad=-slim, w_pad=-slim, h_pad=-slim)
plt.savefig(gp_fig_folder + "plot.pdf",
            bbox_inches="tight")
plt.show()

# %% [markdown]
# # B.2

# %% [markdown]
# ## Algorithm definition

# %%
def algorithm1(X, y, XNew, T, C=C, W=W, N = N, prior_dict = None, kernel = None):
    X_aug = torch.cat((X, torch.empty(T)))
    y_aug = torch.cat((y, torch.empty(T)))
    X_dim = X.shape[0]
    y_dim = y.shape[0]
    stats = torch.empty((T, 3, XNew.shape[0]))
    minima = torch.empty((T, 2))
    mcmc = None
    #clear_output(wait=True)
    for k in range(T):
        pyro.clear_param_store()
        print("Iteration " + str(k+1) + "/" + str(T))
        X_k = X_aug[:X_dim + k]
        y_k = y_aug[:y_dim + k]
        gpr = make_gpr(X_k, y_k, prior_dict = prior_dict, kernel = kernel)
        mcmc = mcmc_sampler(gpr, C, W, N)
        posterior_predictive, _ = sample_predict(XNew, gpr, mcmc, prior_dict = prior_dict)
        fs = posterior_predictive['f'].mean(dim=0)
        ps = torch.argmin(fs)
        X_min = XNew[ps]
        y_min = f(X_min)
        X_aug[X_dim+k] = X_min
        y_aug[y_dim+k] = y_min
        print(posterior_predictive['loc'].mean(dim=0).shape)
        stats[k, 0, :] = posterior_predictive['loc'].mean(dim=0)
        stats[k, 1, :] = posterior_predictive['var'].mean(dim=0)
        stats[k, 2, :] = fs
        minima[k, 0] = X_min
        minima[k, 1] = y_min
    return stats, minima, mcmc

# %% [markdown]
# ## Algorithm hyper-parameters

# %%
T = 10
C = 1
N = 100
W = 100

# %% [markdown]
# ## B.2-1 plotting

# %% [markdown]
# ## Plot for $k \in \{1, 5, 10\}$

# %%
stats, minima, mcmc = algorithm1(X, y, XNew, T, C, W, N)

def plot_b2(X, XNew, y, stats, minima, k = 0, save_prefix = '', save=False):
    plt.title(r"k = {}".format(k+1))
    plt.plot(XNew, f(XNew), label = r"$f(x)$")
    plt.plot(XNew, stats[k,2,:], label= r"$f^*$")
    plt.plot(XNew, stats[k,0,:] + 2 * np.sqrt(stats[k,1,:]), label=r"$m(x^*) + 2\sqrt{v(x^*)}$", linestyle='dotted', color='tab:purple')
    plt.plot(XNew, stats[k,0,:] - 2 * np.sqrt(stats[k,1,:]), label=r"$m(x^*) - 2\sqrt{v(x^*)}$", linestyle='dashed', color='tab:purple')
    plt.scatter(X, y, color = 'tab:red', label = r"$\mathcal{D}$", zorder=4)
    plt.scatter(minima[k,0], minima[k,1], color = 'tab:green', label = r"$(x^*_p, f(x^*_p))$", zorder=4)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend()
    slim=0.6
    plt.tight_layout(pad=-slim, w_pad=-slim, h_pad=-slim)
    if save:
        plt.savefig(gp_fig_folder + save_prefix+"k_{}.pdf".format(k+1), bbox_inches="tight")
    plt.show()

for k in [0, 4, 9]:
    plot_b2(X, XNew, y, stats, minima, k)

# %% [markdown]
# ## Trying different kernels / grid search of priors

# %% [markdown]
# ### For reiteration, default parameters used for b2

# %%
# prior given in assignment text
prior_dict = {}
prior_dict['variance'] = pyro.nn.PyroSample(dist.LogNormal(0., 1.5))
prior_dict['lengthscale'] = pyro.nn.PyroSample(dist.LogNormal(-1., 1.))
kernel = gp.kernels.RBF(input_dim=1)

stats, minima, mcmc = algorithm1(X, y, XNew, T, C, W, N, prior_dict=prior_dict, kernel=kernel)
for k in [0, 4, 9]:
    plot_b2(X, XNew, y, stats, minima, k)

# %% [markdown]
# ### Brownian kernel

# %%
prior_dict = {}
prior_dict['variance'] = pyro.nn.PyroSample(dist.LogNormal(0., 2.))
kernel = gp.kernels.Brownian(input_dim=1)
stats, minima, mcmc = algorithm1(X, y, XNew, T, C, W, N, prior_dict=prior_dict, kernel=kernel)
for k in [0, 4, 9]:
    plot_b2(X, XNew, y, stats, minima, k)

# %% [markdown]
# ### Matern32 kernel

# %%
prior_dict = {}
prior_dict['variance'] = pyro.nn.PyroSample(dist.LogNormal(0., 2.))
prior_dict['lengthscale'] = pyro.nn.PyroSample(dist.LogNormal(-1., 1.))
kernel = gp.kernels.Matern32(input_dim=1)
stats, minima, mcmc = algorithm1(X, y, XNew, T, C, W, N, prior_dict=prior_dict, kernel=kernel)
for k in [0, 4, 9]:
    plot_b2(X, XNew, y, stats, minima, k)


# %% [markdown]
# ### Cosine kernel

# %%
prior_dict = {}
prior_dict['variance'] = pyro.nn.PyroSample(dist.LogNormal(0., 2.))
prior_dict['lengthscale'] = pyro.nn.PyroSample(dist.LogNormal(-1., 1.))
kernel = gp.kernels.Cosine(input_dim = 1)
stats, minima, mcmc = algorithm1(X, y, XNew, T, C, W, N, prior_dict=prior_dict, kernel=kernel)
for k in [0, 4, 9]:
    plot_b2(X, XNew, y, stats, minima, k)


