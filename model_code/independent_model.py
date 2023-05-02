
import numpyro
from numpyro.distributions import constraints
from jax import numpy as jnp
from jax import random
import arviz as az

from scipy.stats import linregress

from numpyro.infer import MCMC, NUTS, Predictive

# import pyro
# import pyro.distributions as dist
# import pyro.poutine as poutine
# from pyro.infer import Predictive

import torch
import time
import pandas as pd
import numpy as np

# from pyro.infer import SVI, Trace_ELBO
#
# import torch.distributions.constraints as constraints

from simulation_code import DataSimulator
# from utils import compare_to_gt

import pickle

def two_nodes_run_level(data, missing, priors=None):

    ## Initialize experiment wide params
    beta0 = numpyro.sample("beta0", numpyro.distributions.Normal(4., 1.))
    beta1 = numpyro.sample("beta1", numpyro.distributions.Normal(.5, .25))
    mar = numpyro.sample("mar", numpyro.distributions.LogNormal(-3.5, .1))


    ## TODO: BAD VECTORIZE
    # with numpyro.plate("Protein_plate", len(data)):
    run_mu_list = numpyro.sample("mu", numpyro.distributions.Normal(priors["run_effect"], 2.))

    ## Feature means
    feature_mu_list = numpyro.sample("bF", numpyro.distributions.Normal(priors["feature_effect"], 1.))

    ## Error
    sigma = numpyro.sample("error", numpyro.distributions.Exponential(1.))

    run_mu = run_mu_list[np.arange(run_mu_list.shape[0])[:, None], data[:,:, 0].astype(int)]
    feature_mu = feature_mu_list[np.arange(run_mu_list.shape[0])[:, None], data[:,:, 1].astype(int)]

    mnar_missing = 1 / (1 + jnp.exp(-beta0 + (beta1 * (run_mu + feature_mu)) - (.5*beta1*sigma)))
    mnar_not_missing = 1 / (1 + jnp.exp(-beta0 + (beta1 * (run_mu + feature_mu))))
    mnar = jnp.where(missing, mnar_missing, mnar_not_missing)

    missing_prob = mar + ((1 - mar) * mnar)

    numpyro.distributions.constraints.positive(numpyro.sample("missing",
                          numpyro.distributions.Bernoulli(probs=missing_prob),
                          obs=missing))
    adjustment = mnar/(mar+mnar)*(.5 * beta1 * sigma)

    imp_means = (run_mu + feature_mu - adjustment).flatten()
    imp = numpyro.sample(
        "imp", numpyro.distributions.Normal(
            imp_means[missing.flatten()],
            sigma).mask(False)
    )

    obs = data[:, :, 2]
    ## replace indices in a 2d numpy array with those in another 2d array
    observed = jnp.asarray(obs).at[missing].set(imp)

    mean = jnp.where(missing,
                     run_mu + feature_mu - adjustment,
                     run_mu + feature_mu)

    numpyro.sample("obs", numpyro.distributions.Normal(mean, sigma), obs=observed)

class IndependentModel:

    def __init__(self):
        self.priors = None

        pass

    def format_data(self, data):

        input_data = list()
        model_params = dict()

        formatted_data = data.loc[:, ["Protein", "Condition", "Run", "Feature", "Intensity", "Missing"]]
        for col in ["Protein", "Condition", "Run"]:
            if formatted_data[col].dtype == 'O':
                formatted_data.loc[:, col] = formatted_data.loc[:, col].astype("category").cat.codes

        formatted_data.loc[:, "Condition_run"] = formatted_data.loc[:, "Condition"].astype(str) + "_" + \
                                                 formatted_data.loc[:, "Run"].astype(str)
        formatted_data.loc[:, "Run"] = formatted_data.loc[:, "Condition_run"].astype("category").cat.codes
        formatted_data = formatted_data.drop(columns=["Condition"])
        # formatted_data.loc[:, "Dummy_Condition"] = formatted_data.loc[:, "Condition"].astype(str)
        # formatted_data.loc[:, "Dummy_Run"] = formatted_data.loc[:, "Run"].astype(str)
        # formatted_data.loc[:, "Dummy_Feature"] = formatted_data.loc[:, "Feature"].astype(str)
        for i in formatted_data["Protein"].unique():

            temp_data = formatted_data.loc[formatted_data["Protein"] == i]
            if temp_data["Feature"].dtype == 'O':
                temp_data.loc[:, "Feature"] = temp_data.loc[:, "Feature"].astype("category").cat.codes

            temp_data = pd.get_dummies(temp_data, drop_first=False)
            temp_data.loc[:, "Missing"] = np.where(np.isnan(temp_data["Intensity"]), 1., 0.)
            # n_conds = len(temp_data.columns[temp_data.columns.str.contains("Dummy_Condition")])
            # n_runs = len(temp_data.columns[temp_data.columns.str.contains("Dummy_Run")])
            n_runs = len(temp_data.loc[:, "Run"].unique())
            n_feat = len(temp_data.loc[:, "Feature"].unique())
            # n_feat = len(temp_data.columns[temp_data.columns.str.contains("Dummy_Feature")])
            model_params[i] = {"Runs" : n_runs,
                               "Features" : n_feat}
            temp_data = jnp.asarray(temp_data.drop(columns="Protein").values)
            temp_data = np.nan_to_num(temp_data, nan=0.)
            input_data.append(temp_data)

        input_data = np.array(input_data)

        return input_data, model_params

    def get_priors(self, data):

        ## Initialize collection structures
        priors = dict()
        run_priors = list()
        feature_priors = list()

        for i in range(len(data)):
            # conditions = len(np.unique(data[i][:, 0]))
            runs = len(np.unique(data[i][:, 0]))
            features = len(np.unique(data[i][:, 1]))

            ## Overall mean
            overall_mean = data[i][:, 2][data[i][:, 2] != 0].mean()

            # ## Calculate condition priors
            # condition_effect = list()
            # condition_std = list()
            #
            # for c in range(conditions):
            #     condition_effect.append(data[i][:, 3][(data[i][:, 0] == c) & (data[i][:, 3] != 0)].mean() - overall_mean)
            #     condition_std.append(data[i][:, 3][(data[i][:, 0] == c) & (data[i][:, 3] != 0)].std())

            # condition_effect = jnp.array(condition_effect)
            # condition_effect = np.nan_to_num(condition_effect, nan=0.)
            # condition_std = jnp.array(condition_std)
            # condition_std = np.nan_to_num(condition_std, nan=1.)

            ## Calculate run priors
            run_effect = list()
            run_std = list()

            for r in range(runs):
                run_effect.append(data[i][:, 2][(data[i][:, 0] == r) & (data[i][:, 2] != 0)].mean())
                run_std.append(data[i][:, 2][(data[i][:, 0] == r) & (data[i][:, 2] != 0)].std())

            run_effect = np.array(run_effect)
            run_effect = np.nan_to_num(run_effect, nan=0.)
            run_priors.append(run_effect)

            # run_std = np.array(run_std)
            # run_std = np.nan_to_num(run_std, nan=1.)

            ## Calculate feature priors
            feature_effect = list()
            feature_std = list()

            for f in range(features):
                feature_effect.append(data[i][:, 2][(data[i][:, 1] == f) & (data[i][:, 2] != 0)].mean() - overall_mean)
                feature_std.append(data[i][:, 2][(data[i][:, 1] == f) & (data[i][:, 2] != 0)].std())

            feature_effect = np.array(feature_effect)
            feature_effect = np.nan_to_num(feature_effect, nan=0.)
            feature_priors.append(feature_effect)
            # feature_std = jnp.array(feature_std)
            # feature_std = np.nan_to_num(feature_std, nan=1.)

            # priors["overall_mean_{}".format(i)] = overall_mean
            # priors["condition_effect_{}".format(i)] = condition_effect
            # priors["condition_std_{}".format(i)] = condition_std
            # priors["run_effect_{}".format(i)] = run_effect
            # priors["run_std_{}".format(i)] = run_std
            # priors["feature_effect_{}".format(i)] = feature_effect
            # priors["feature_std_{}".format(i)] = feature_std

        priors = dict()
        run_priors = np.array(run_priors)
        feature_priors = np.array(feature_priors)
        priors["run_effect"] = run_priors
        priors["feature_effect"] = feature_priors

        self.priors = priors

    def train(self, data, warmup_steps, sample_steps):
        start = time.time()
        format_data, params = self.format_data(data)

        missing = list()
        for i in range(len(format_data)):
            missing.append(format_data[i][:, 3] == 1.)
        missing = np.array(missing)

        self.get_priors(format_data)
        numpyro.set_host_device_count(4)
        mcmc = MCMC(NUTS(two_nodes_run_level), num_warmup=warmup_steps, num_samples=sample_steps, num_chains=1)#

        mcmc.run(random.PRNGKey(69), format_data, missing, priors=self.priors)
        finish = time.time()
        print("Time to train: {}".format(finish - start))
        keep = finish-start

        print(mcmc.print_summary())

        idata = az.from_numpyro(mcmc)
        with open(r"data/real_data_20_proteins.pickle", "wb") as output_file:
            pickle.dump(idata, output_file)


def main():
    with open(r"data/simulated_data_25.pickle", "rb") as input_file:
        simulator = pickle.load(input_file)
    input_data = simulator.data
    # input_data = pd.read_csv(r"data/Choi2017_model_input.csv")
    # sample_proteins = np.random.choice(input_data["Protein"].unique(), 100)
    # # sample_proteins = ['P47133', 'P40078', 'Q03327', 'P38803', 'P38255', 'Q12402',
    # #                    'P40302', 'P36120', 'P36141', 'P07283',
    # #                    'P35724', 'Q01852', 'Q02159', 'Q01662', 'P38758', 'P38889',
    # #                    'P40024', 'P53741', 'P40040', 'P50087', 'P25617', 'P54839',
    # #                    'P53301', 'P38332', 'P31383', 'P08964', 'P53163', 'Q04693',## Works
    # #                    'P40015', 'Q02455', 'P38882', 'P36154', 'P20606', 'Q12198',
    # #                    'Q06142', 'P50101', 'P40989', 'Q06682', 'P32465', 'P47031',
    # #                    'P38708', 'P34232', 'P39743', 'P35179', 'Q02256', 'Q06708']
    # # print(sample_proteins)
    # input_data = input_data.loc[input_data["Protein"].isin(sample_proteins)]
    # input_data = input_data.loc[input_data["Protein"].isin(["P40527", "P48363"])]
    model = IndependentModel()
    model.train(input_data, 10000, 10000)

if __name__ == "__main__":
    main()

# def two_nodes_run_level(data, missing, runs=50, features=50, learn_priors_from_data=True,
#                         priors=None):
#     ## Initialize model
#     # nodes = list(nx.topological_sort(network))
#
#     # beta0 = 6.5
#     # beta1 = .4
#     beta0 = numpyro.sample("beta0", numpyro.distributions.Normal(10., 5.))
#     beta1 = numpyro.sample("beta1", numpyro.distributions.Normal(1., .25))
#     mar = numpyro.sample("mar", numpyro.distributions.LogNormal(-3, .1))
#
#     run_mu_list = list()
#     feature_mu_list = list()
#     sigma_list = list()
#
#     # if learn_priors_from_data:
#     #
#     #     for i in range(len(data)):
#     #
#     #         run_mu_list.append(numpyro.sample("mu_{}".format(i),
#     #                                           numpyro.distributions.Normal(priors["run_effect_{}".format(i)], 10.)))
#     #
#     #         ## Feature means
#     #         feature_mu_list.append(numpyro.sample("bF_{}".format(i),
#     #                                               # numpyro.distributions.Normal(0., 2.).expand([features])))
#     #                                               numpyro.distributions.Normal(priors["feature_effect_{}".format(i)],
#     #                                                                            1.)))
#     #
#     #         ## Error
#     #         sigma_list.append(numpyro.sample("error_{}".format(i), numpyro.distributions.Exponential(1.)))
#     #
#     # else:
#     #     for i in range(len(data)):
#     #         run_mu_list.append(numpyro.sample("mu_{}".format(i), numpyro.distributions.Uniform(10, 20).expand([runs])))
#     #
#     #         ## Feature means
#     #         feature_mu_list.append(numpyro.sample("bF_{}".format(i),
#     #                                               numpyro.distributions.Normal(0., 2.).expand([features])))
#     #
#     #         sigma_list.append(numpyro.sample("error_{}".format(i), numpyro.distributions.Uniform(0., 2.)))
#
#     for i in range(len(data)):
#         run_mu_list.append(
#             numpyro.sample("mu_{}".format(i), numpyro.distributions.Normal(priors["run_effect_{}".format(i)], 10.))
#         )
#
#         ## Feature means
#         feature_mu_list.append(
#             numpyro.sample("bF_{}".format(i), numpyro.distributions.Normal(priors["feature_effect_{}".format(i)], 1.))
#         )
#
#         ## Error
#         sigma_list.append(
#             numpyro.sample("error_{}".format(i), numpyro.distributions.Exponential(1.))
#         )
#
#         run_mu_temp = run_mu_list[i][data[i][:, 0].astype(int)]
#         feature_mu_temp = feature_mu_list[i][data[i][:, 1].astype(int)]
#         sigma_temp = sigma_list[i]
#
#         temp_missing = missing[i]
#
#         mnar_missing = 1 / (1 + jnp.exp(-beta0 + (beta1 * (run_mu_temp + feature_mu_temp)) - (.5*beta1*sigma_temp)))
#         mnar_not_missing = 1 / (1 + jnp.exp(-beta0 + (beta1 * (run_mu_temp + feature_mu_temp))))
#         mnar = jnp.where(temp_missing == 1, mnar_missing, mnar_not_missing)
#
#         missing_prob = mar + ((1 - mar) * mnar)
#
#         numpyro.distributions.constraints.positive(numpyro.sample("missing_{}".format(i),
#                               numpyro.distributions.Bernoulli(probs=missing_prob),
#                               obs=temp_missing))
#         adjustment = mnar/(mar+mnar)*(.5 * beta1 * sigma_temp)#
#
#         imp = numpyro.sample(
#             "imp_{}".format(i), numpyro.distributions.Normal(
#                 run_mu_temp[temp_missing] + feature_mu_temp[temp_missing] - adjustment[temp_missing],
#                 sigma_temp).mask(False)
#         )
#
#         obs = data[i][:, 2]
#         observed = jnp.asarray(obs).at[temp_missing].set(imp)
#         mean = jnp.where(temp_missing==1,
#                          run_mu_temp + feature_mu_temp - adjustment,
#                          run_mu_temp + feature_mu_temp)
#
#         numpyro.sample("obs_{}".format(i), numpyro.distributions.Normal(mean, sigma_temp), obs=observed)