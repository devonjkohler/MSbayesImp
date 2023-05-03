
# !/usr/bin/env python3.7
import sys
import builtins
import os
sys.stdout = open("stdout.txt", "w", buffering=1)
def print(text):
    builtins.print(text)
    os.fsync(sys.stdout)

print("This is immediately written to stdout.txt")

import numpyro
from numpyro.infer import MCMC, NUTS
numpyro.set_platform('gpu')
# numpyro.set_host_device_count(2)

import jax
from jax import numpy as jnp
from jax import random
# import az.from_numpyro as az_numpy
# import arviz as az

import time
import pickle

import pandas as pd
import numpy as np

from simulation_code import DataSimulator

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
print(jax.local_device_count())

def two_nodes_run_level(data, missing, priors=None):

    ## Initialize experiment wide params
    beta0 = numpyro.sample("beta0", numpyro.distributions.Normal(4., 1.))
    beta1 = numpyro.sample("beta1", numpyro.distributions.Normal(.5, .25))
    mar = numpyro.sample("mar", numpyro.distributions.LogNormal(-3.5, .1))

    ## Initialize model variables
    run_mu_list = numpyro.sample("mu", numpyro.distributions.Normal(priors["run_effect"], 2.))
    feature_mu_list = numpyro.sample("bF", numpyro.distributions.Normal(priors["feature_effect"], 1.))
    sigma = numpyro.sample("error", numpyro.distributions.Exponential(1.))

    ## Get model param for each obs
    # run_mu = run_mu_list[np.arange(run_mu_list.shape[0])[:, None], data[:,:, 0].astype(int)]
    # feature_mu = feature_mu_list[np.arange(run_mu_list.shape[0])[:, None], data[:,:, 1].astype(int)]
    run_mu = run_mu_list[data[:, 0].astype(int)]
    feature_mu = feature_mu_list[data[:, 1].astype(int)]

    ## Calculate missingness probability
    mnar_missing = 1 / (1 + jnp.exp(-beta0 + (beta1 * (run_mu + feature_mu)) - (.5*beta1*sigma)))
    mnar_not_missing = 1 / (1 + jnp.exp(-beta0 + (beta1 * (run_mu + feature_mu))))
    mnar = jnp.where(missing, mnar_missing, mnar_not_missing)

    missing_prob = mar + ((1 - mar) * mnar)

    numpyro.distributions.constraints.positive(numpyro.sample("missing",
                          numpyro.distributions.Bernoulli(probs=missing_prob),
                          obs=missing))

    ## Infer missing values
    adjustment = mnar/(mar+mnar)*(.5 * beta1 * sigma)

    imp_means = run_mu + feature_mu - adjustment
    imp = numpyro.sample(
        "imp", numpyro.distributions.Normal(
            imp_means[missing==1],
            sigma).mask(False)
    )

    ## Add imputed missing values to observations
    obs = data[:, 2]
    observed = jnp.asarray(obs).at[missing==1].set(imp)

    ## Sample with obs
    mean = jnp.where(missing==1,
                     run_mu + feature_mu - adjustment,
                     run_mu + feature_mu)

    numpyro.sample("obs", numpyro.distributions.Normal(mean, sigma), obs=observed)

class IndependentModel:

    def __init__(self):
        self.priors = None

        pass

    def format_data(self, data):

        formatted_data = data.loc[:, ["Protein", "Condition", "Run", "Feature", "Intensity", "Missing"]]
        for col in ["Protein", "Condition", "Run", "Feature"]:
            if formatted_data[col].dtype == 'O':
                formatted_data.loc[:, col] = formatted_data.loc[:, col].astype("category").cat.codes

        formatted_data.loc[:, "Condition_run"] = formatted_data.loc[:, "Condition"].astype(str) + "_" + \
                                                 formatted_data.loc[:, "Run"].astype(str)
        formatted_data.loc[:, "Run"] = formatted_data.loc[:, "Condition_run"].astype("category").cat.codes
        formatted_data = formatted_data.drop(columns=["Condition", "Condition_run"])

        formatted_data.loc[:, "Missing"] = np.where(np.isnan(formatted_data["Intensity"]), 1., 0.)
        formatted_data = formatted_data.sort_values(by=["Protein", "Feature", "Run"])

        formatted_data = formatted_data.fillna(0.)

        return formatted_data.values

    def flatten_input(self, data):

        data = pd.DataFrame(data, columns=["Protein", "Run", "Feature", "Intensity", "Missing"])

        ## Format input data to be ready for flatten
        data.loc[:, "list_index"] = np.arange(len(data))
        data.loc[:, "Protein_run"] = data.loc[:, "Protein"].astype(str) + "_" + data.loc[:, "Run"].astype(str)
        data.loc[:, "Protein_feature"] = data.loc[:, "Protein"].astype(str) + "_" + data.loc[:, "Feature"].astype(str)

        ## TODO: This is sorta gross but cat.codes doesn't create these in order
        data = pd.merge(data, pd.DataFrame({"Protein_run" : data.loc[:, "Protein_run"].unique(),
                      "Protein_run_idx" : np.arange(len(data.loc[:, "Protein_run"].unique()))}),
                 on="Protein_run", how="left")
        data = pd.merge(data, pd.DataFrame({
            "Protein_feature": data.loc[:, "Protein_feature"].unique(),
            "Protein_feature_idx": np.arange(len(data.loc[:, "Protein_feature"].unique()))}),
                 on="Protein_feature", how="left")

        ## Return data for model and interpretation
        flatten_data = data.loc[:, ["Protein_run_idx", "Protein_feature_idx", "Intensity"]].values
        self.flat_input = flatten_data
        self.lookup_table = data

    def get_priors(self, data):

        ## Initialize collection structures
        run_priors = list()
        feature_priors = list()

        proteins = np.unique(data[:, 0])

        for i in range(len(proteins)):

            temp_data = data[data[:, 0] == proteins[i]]

            runs = np.unique(temp_data[:, 1])
            features = np.unique(temp_data[:, 2])

            ## Overall mean
            overall_mean = temp_data[:, 3][temp_data[:, 3] != 0].mean()

            ## Calculate run priors
            run_effect = list()
            run_std = list()

            for r in runs:
                run_effect.append(temp_data[:, 3][(temp_data[:, 1] == r) & (temp_data[:, 3] != 0)].mean())
                run_std.append(temp_data[:, 3][(temp_data[:, 1] == r) & (temp_data[:, 3] != 0)].std())

            run_effect = np.array(run_effect)
            run_effect = np.nan_to_num(run_effect, nan=0.)
            run_priors.append(run_effect)

            ## Calculate feature priors
            feature_effect = list()
            feature_std = list()

            for f in features:
                feature_effect.append(temp_data[:, 3][(temp_data[:, 2] == f) &
                                                      (temp_data[:, 3] != 0)].mean() - overall_mean)
                feature_std.append(temp_data[:, 3][(temp_data[:, 2] == f) & (temp_data[:, 3] != 0)].std())

            feature_effect = np.array(feature_effect)
            feature_effect = np.nan_to_num(feature_effect, nan=0.)
            feature_priors.append(feature_effect)

        priors = dict()
        run_priors = np.concatenate(run_priors)
        feature_priors = np.concatenate(feature_priors)
        priors["run_effect"] = run_priors
        priors["feature_effect"] = feature_priors

        self.priors = priors

    def train(self, data, warmup_steps, sample_steps):
        start = time.time()

        ## Format data for training
        format_data = self.format_data(data)
        missing = format_data[:, 4]

        ## Get priors
        self.get_priors(format_data)
        self.flatten_input(format_data)

        mcmc = MCMC(NUTS(two_nodes_run_level), num_warmup=warmup_steps, num_samples=sample_steps, num_chains=1)#

        mcmc.run(random.PRNGKey(69), self.flat_input, missing, priors=self.priors)
        finish = time.time()
        keep = finish-start

        print(mcmc.print_summary())
        print("Time to train: {}".format(keep))

        # idata = az.from_numpyro(mcmc)
        # self.az_mcmc_results = idata
        self.mcmc_samples = mcmc.get_samples()

def main():
    # with open(r"data/simulated_data_5.pickle", "rb") as input_file:
    #     simulator = pickle.load(input_file)
    # input_data = simulator.data
    # input_data = pd.read_csv(r"/home/kohler.d/MSbayesImp/model_code/data/sim_data.csv")
    input_data = pd.read_csv(r"/home/kohler.d/MSbayesImp/model_code/data/Choi2017_model_input.csv")
    sample_proteins = np.random.choice(input_data["Protein"].unique(), 1000)
    input_data = input_data.loc[input_data["Protein"].isin(sample_proteins)]

    model = IndependentModel()
    model.train(input_data, 15**4, 15**4)
    with open(r"/scratch/kohler.d/real_samples.pickle", "wb") as output_file:
        pickle.dump(model.mcmc_samples, output_file)

    with open(r"/scratch/kohler.d/real_lookup_table.pickle", "wb") as output_file:
        pickle.dump(model.lookup_table, output_file)

    # with open(r"model_results/az_mcmc_results.pickle", "wb") as output_file:
    #     pickle.dump(model.az_mcmc_results, output_file)

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