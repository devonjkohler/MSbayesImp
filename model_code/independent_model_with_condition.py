
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

import pandas as pd
import numpy as np

# from pyro.infer import SVI, Trace_ELBO
#
# import torch.distributions.constraints as constraints

from simulation_code import DataSimulator
# from utils import compare_to_gt

import pickle

def two_nodes_run_level(data, missing, runs=50, features=50, learn_priors_from_data=True,
                        priors=None):

    # beta0 = 6.5
    # beta1 = .4
    beta0 = numpyro.sample("beta0", numpyro.distributions.Normal(10., 1.))
    beta1 = numpyro.sample("beta1", numpyro.distributions.Normal(1., .1))
    mar = numpyro.sample("mar", numpyro.distributions.LogNormal(-3, .2))

    mu_list = list()
    condition_mu_list = list()
    run_mu_list = list()
    feature_mu_list = list()
    sigma_list = list()

    if learn_priors_from_data:

        for i in range(len(data)):
            mu_list.append(numpyro.sample("mu_{}".format(i),
                                          numpyro.distributions.Normal(priors["overall_mean_{}".format(i)], 10.)))

            ## Condition means
            condition_mu_list.append(numpyro.sample("bC_{}".format(i),
                                         numpyro.distributions.Normal(priors["condition_effect_{}".format(i)], 3.)))

            ## Run means
            run_mu_list.append(numpyro.sample("bR_{}".format(i),
                                              numpyro.distributions.Normal(priors["run_effect_{}".format(i)], 3.)))

            ## Feature means
            feature_mu_list.append(numpyro.sample("bF_{}".format(i),
                                                  numpyro.distributions.Normal(priors["feature_effect_{}".format(i)],
                                                                               1.)))

            ## Error
            sigma_list.append(numpyro.sample("error_{}".format(i), numpyro.distributions.Exponential(1.)))

    else:
        for i in range(len(data)):
            run_mu_list.append(numpyro.sample("mu_{}".format(i), numpyro.distributions.Uniform(10, 20).expand([runs])))

            ## Feature means
            feature_mu_list.append(numpyro.sample("bF_{}".format(i),
                                                  numpyro.distributions.Normal(0., 2.).expand([features])))

            sigma_list.append(numpyro.sample("error_{}".format(i), numpyro.distributions.Uniform(0., 2.)))

    for i in range(len(data)):
        mu_temp = mu_list[i]
        condition_temp = condition_mu_list[i][data[i][:, 0].astype(int)]
        run_mu_temp = run_mu_list[i][data[i][:, 1].astype(int)]
        feature_mu_temp = feature_mu_list[i][data[i][:, 2].astype(int)]
        sigma_temp = sigma_list[i]

        temp_missing = missing[i]

        mean = mu_temp + condition_temp + run_mu_temp + feature_mu_temp

        mnar_missing = 1 / (1 + jnp.exp(-beta0 + (beta1 * mean) - (.5*beta1*sigma_temp)))
        mnar_not_missing = 1 / (1 + jnp.exp(-beta0 + (beta1 * mean)))
        mnar = jnp.where(temp_missing == 1, mnar_missing, mnar_not_missing)

        missing_prob = mar + ((1 - mar) * mnar)

        numpyro.distributions.constraints.positive(numpyro.sample("missing_{}".format(i),
                              numpyro.distributions.Bernoulli(probs=missing_prob),
                              obs=temp_missing))
        adjustment = mnar/(mar+mnar)*(.5 * beta1 * sigma_temp)

        imp = numpyro.sample(
            "imp_{}".format(i), numpyro.distributions.Normal(
                mu_temp + condition_temp[temp_missing] + run_mu_temp[temp_missing] + \
                                feature_mu_temp[temp_missing] - adjustment[temp_missing],
                sigma_temp).mask(False)
        )

        obs = data[i][:, 3]
        observed = jnp.asarray(obs).at[temp_missing].set(imp)
        final_mean = jnp.where(temp_missing==1,
                               mu_temp + condition_temp + run_mu_temp + feature_mu_temp - adjustment,
                               mu_temp + condition_temp + run_mu_temp + feature_mu_temp)

        numpyro.sample("obs_{}".format(i), numpyro.distributions.Normal(final_mean, sigma_temp), obs=observed)

class IndependentModel:

    def __init__(self):
        self.priors = None

        pass

    def format_data(self, data):

        input_data = dict()
        model_params = dict()

        formatted_data = data.loc[:, ["Protein", "Condition", "Run", "Feature", "Intensity", "Missing"]]
        formatted_data.loc[:, "Dummy_Condition"] = formatted_data.loc[:, "Condition"].astype(str)
        formatted_data.loc[:, "Dummy_Run"] = formatted_data.loc[:, "Run"].astype(str)
        formatted_data.loc[:, "Dummy_Feature"] = formatted_data.loc[:, "Feature"].astype(str)

        for i in formatted_data["Protein"].unique():
            temp_data = formatted_data.loc[formatted_data["Protein"] == i]
            temp_data = pd.get_dummies(temp_data, drop_first=False)
            temp_data.loc[:, "Missing"] = np.where(np.isnan(temp_data["Intensity"]), 1., 0.)
            n_conds = len(temp_data.columns[temp_data.columns.str.contains("Dummy_Condition")])
            n_runs = len(temp_data.columns[temp_data.columns.str.contains("Dummy_Run")])
            n_feat = len(temp_data.columns[temp_data.columns.str.contains("Dummy_Feature")])
            model_params[i] = {"Conditions": n_conds,
                               "Runs" : n_runs,
                               "Features" : n_feat}
            temp_data = jnp.asarray(temp_data.drop(columns="Protein").values)
            temp_data = np.nan_to_num(temp_data, nan=0.)
            input_data[i] = temp_data

        return input_data, model_params

    def get_priors(self, data):

        priors = dict()

        for i in range(len(data)):
            conditions = len(np.unique(data[i][:, 0]))
            runs = len(np.unique(data[i][:, 1]))
            features = len(np.unique(data[i][:, 2]))

            ## Overall mean
            overall_mean = data[i][:, 3][data[i][:, 3] != 0].mean()

            ## Calculate condition priors
            condition_effect = list()
            condition_std = list()

            for c in range(conditions):
                condition_effect.append(data[i][:, 3][(data[i][:, 0] == c) & (data[i][:, 3] != 0)].mean() - overall_mean)
                condition_std.append(data[i][:, 3][(data[i][:, 0] == c) & (data[i][:, 3] != 0)].std())

            condition_effect = jnp.array(condition_effect)
            condition_effect = np.nan_to_num(condition_effect, nan=0.)
            condition_std = jnp.array(condition_std)
            condition_std = np.nan_to_num(condition_std, nan=1.)

            ## Calculate run priors
            run_effect = list()
            run_std = list()

            for r in range(runs):
                run_effect.append(data[i][:, 3][(data[i][:, 1] == r) & (data[i][:, 3] != 0)].mean() - overall_mean)
                run_std.append(data[i][:, 3][(data[i][:, 1] == r) & (data[i][:, 3] != 0)].std())

            run_effect = jnp.array(run_effect)
            run_effect = np.nan_to_num(run_effect, nan=0.)
            run_std = jnp.array(run_std)
            run_std = np.nan_to_num(run_std, nan=1.)

            ## Calculate feature priors
            feature_effect = list()
            feature_std = list()

            for f in range(features):
                feature_effect.append(data[i][:, 3][(data[i][:, 2] == f) & (data[i][:, 3] != 0)].mean() - overall_mean)
                feature_std.append(data[i][:, 3][(data[i][:, 2] == f) & (data[i][:, 3] != 0)].std())

            feature_effect = jnp.array(feature_effect)
            feature_effect = np.nan_to_num(feature_effect, nan=0.)
            feature_std = jnp.array(feature_std)
            feature_std = np.nan_to_num(feature_std, nan=1.)

            priors["overall_mean_{}".format(i)] = overall_mean
            priors["condition_effect_{}".format(i)] = condition_effect
            priors["condition_std_{}".format(i)] = condition_std
            priors["run_effect_{}".format(i)] = run_effect
            priors["run_std_{}".format(i)] = run_std
            priors["feature_effect_{}".format(i)] = feature_effect
            priors["feature_std_{}".format(i)] = feature_std

        self.priors = priors

    def train(self, data, num_iters, initial_lr, gamma):

        format_data, params = self.format_data(data)

        missing = list()
        for i in range(len(format_data)):
            missing.append(format_data[i][:, 4] == 1.)

        self.get_priors(format_data)

        mcmc = MCMC(NUTS(two_nodes_run_level), num_warmup=7500, num_samples=7500)
        mcmc.run(random.PRNGKey(69), format_data, missing, priors=self.priors)
        print(mcmc.print_summary())

        idata = az.from_numpyro(mcmc)
        with open(r"data/az_mcmc_ind_two.pickle", "wb") as output_file:
            pickle.dump(idata, output_file)


def main():
    with open(r"data/simulated_data_three.pickle", "rb") as input_file:
        simulator = pickle.load(input_file)

    model = IndependentModel()
    model.train(simulator.data, 1000, .01, .01)

    # simulator.data.loc[:, "Protein"] = np.where(simulator.data.loc[:, "Protein"]==0., "A", "B")
    #
    # proteins = ["A", "B"]
    # result_list = list()
    # for i in proteins:
    #     try:
    #         run_effect = model.model_results['br_loc{0}'.format(i)]
    #         # try:
    #         #     coef_effect = model.model_results["b{0}_loc".format(i)]
    #         #     run_mean = temp_mean + model.model_results['mu_locA']*coef_effect + run_effect
    #         #     # run_mean = torch.cat((temp_mean.reshape(1) + model.model_results['mu_locA']*coef_effect, run_mean))
    #         # except:
    #         if i == "A":
    #             temp_mean = model.model_results['mu_loc{0}'.format(i)]
    #         elif i == "B":
    #             temp_mean = model.model_results['mu_locB'] + \
    #                         ((model.model_results['mu_locA'] + model.model_results['br_locA']) *
    #                          model.model_results['bB_loc'])
    #         run_mean = temp_mean + run_effect
    #         # run_mean = torch.cat((temp_mean.reshape(1), run_mean))
    #         tmp_results = pd.DataFrame(data=run_mean.reshape(1, len(run_mean)).detach().numpy(),
    #                                    columns=simulator.data["Run"].unique())
    #         tmp_results.loc[0, "Protein"] = i
    #         result_list.append(tmp_results)
    #     except:
    #         print("Protein {0} Not found in model".format(i))
    # results_df = pd.concat(result_list)
    # comparison = compare_to_gt(results_df, simulator.data)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # print(comparison["full"])
    # print(comparison["differences"])
    # print(comparison["total"])
    # print(comparison["mean"])

if __name__ == "__main__":
    main()


# def generate_model(data, graph):
#     ordered_nodes = [i for i in graph.topological_sort()]
#     undirected_edges = graph.undirected.edges()
#
#     beta_coefficients = dict()
#     confounder_coefficients = dict()
#     sigma_coefficients = dict()
#     confounder_sigma_coefficients = dict()
#
#     for i in range(len(ordered_nodes)):
#         ancesters = graph.ancestors_inclusive(ordered_nodes[i])
#         confounders = graph.undirected.edges(ordered_nodes[i])
#
#         sigma_coefficients[ordered_nodes[i]] = pyro.sample(
#             "sigma_{}".format(ordered_nodes[i]), dist.Uniform(0., 5.))
#
#         betas = dict()
#         if len(ancesters) > 1:
#             for upstream in ancesters:
#                 if upstream != ordered_nodes[i]:
#                     betas[upstream] = pyro.sample(
#                         "b_{0}_{1}".format(ordered_nodes[i], upstream),
#                         dist.Normal(0., 1.))
#         beta_coefficients[ordered_nodes[i]] = betas
#         if len(confounders) > 0:
#             for e in confounders:
#                 confounder_coefficients[tuple(sorted(e))] = pyro.sample("b_{0}_{1}".format(ordered_nodes[i],
#                                                                                            tuple(sorted(e))),
#                                                                         dist.Normal(0., 1.))
#
#     for e in undirected_edges:
#         confounder_sigma_coefficients[tuple(sorted(e))] = pyro.sample("sigma_{}".format(tuple(sorted(e))),
#                                                                       dist.Uniform(0., 5.))
#
#     with pyro.plate("data", len(data)):
#         latent_variables = dict()
#         observed_variables = dict()
#
#         for e in undirected_edges:
#             latent_variables[tuple(sorted(e))] = pyro.sample("latent_{}".format(tuple(sorted(e))),
#                                                              dist.Normal(0, confounder_sigma_coefficients[
#                                                                  tuple(sorted(e))]))
#
#         for i in range(len(ordered_nodes)):
#             ancesters = graph.ancestors_inclusive(ordered_nodes[i])
#             confounders = graph.undirected.edges(ordered_nodes[i])
#             temp_mean = 0
#             if len(confounders) > 0:
#                 for e in confounders:
#                     temp_mean += confounder_coefficients[tuple(sorted(e))] * latent_variables[tuple(sorted(e))]
#
#             if len(ancesters) > 1:
#                 for upstream in ancesters:
#                     if upstream != ordered_nodes[i]:
#                         temp_mean += beta_coefficients[ordered_nodes[i]][upstream] * observed_variables[
#                             upstream]
#
#             observed_variables[ordered_nodes[i]] = pyro.sample("obs_{}".format(ordered_nodes[i]),
#                                                                dist.Normal(temp_mean,
#                                                                            sigma_coefficients[
#                                                                                ordered_nodes[i]]),
#                                                                obs=torch.tensor(
#                                                                    data.loc[:, str(ordered_nodes[i])].values))
#
# def generate_guide(data, graph):
#     ordered_nodes = [i for i in graph.topological_sort()]
#     undirected_edges = graph.undirected.edges()
#
#     confounder_sigma_coefficients = dict()
#
#     for i in range(len(ordered_nodes)):
#         ancesters = graph.ancestors_inclusive(ordered_nodes[i])
#         confounders = graph.undirected.edges(ordered_nodes[i])
#
#         temp_sigma_loc = pyro.param("sigma_loc_{}".format(ordered_nodes[i]), torch.tensor(1.),
#                                     constraint=constraints.positive)
#         pyro.sample("sigma_{}".format(ordered_nodes[i]), dist.Normal(temp_sigma_loc, torch.tensor(0.05)))
#
#         if len(ancesters) > 1:
#             for upstream in ancesters:
#                 if upstream != ordered_nodes[i]:
#                     temp_b_loc = pyro.param("b_loc_{0}_{1}".format(ordered_nodes[i], upstream),
#                                             torch.tensor(0.))
#                     pyro.sample("b_{0}_{1}".format(ordered_nodes[i], upstream), dist.Normal(temp_b_loc, .1))
#         if len(confounders) > 0:
#             for e in confounders:
#                 temp_b_loc = pyro.param("b_loc_{0}_{1}".format(ordered_nodes[i],
#                                                                tuple(sorted(e))),
#                                         torch.tensor(0.))
#                 pyro.sample("b_{0}_{1}".format(ordered_nodes[i], tuple(sorted(e))),
#                             dist.Normal(temp_b_loc, 1.))
#
#     for e in undirected_edges:
#         temp_sigma_loc = pyro.param("sigma_loc_{}".format(tuple(sorted(e))), torch.tensor(1.),
#                                     constraint=constraints.positive)
#         confounder_sigma_coefficients[tuple(sorted(e))] = pyro.sample("sigma_{}".format(tuple(sorted(e))),
#                                                                       dist.Normal(temp_sigma_loc,
#                                                                                   torch.tensor(0.05)))
#
#     with pyro.plate("data", len(data)):
#
#         for e in undirected_edges:
#             pyro.sample("latent_{}".format(tuple(sorted(e))),
#                         dist.Normal(0, confounder_sigma_coefficients[tuple(sorted(e))]))
#
# def two_nodes_test(A, B, runs, features):
#
#     beta0 = 6.5
#     beta1 = .4
#
#     ## A
#     mu_A = pyro.sample("mu_A", dist.Uniform(1. ,40.))
#     mu_B = pyro.sample("mu_B", dist.Uniform(1., 40.))
#     sigma_run_A = pyro.sample("sigma_run_A", dist.Uniform(0.,10.))
#     sigma_run_B = pyro.sample("sigma_run_B", dist.Uniform(0., 10.))
#
#     with pyro.plate("runs_param", runs):
#         brA = pyro.sample("bR_A", dist.Normal(torch.zeros(runs), torch.ones(runs)))
#         brB = pyro.sample("bR_B", dist.Normal(torch.zeros(runs), torch.ones(runs)))
#
#
#     ## B
#     bB = pyro.sample("bB", dist.Normal(0., .25))
#     sigma_feature = pyro.sample("sigma_feature", dist.Uniform(0.,10.))
#     #
#     with pyro.plate("features_param", features):
#         bfA = pyro.sample("bF_A", dist.Normal(torch.zeros(features), torch.ones(features)))
#         bfB = pyro.sample("bF_B", dist.Normal(torch.zeros(features), torch.ones(features)))
#
#
#     A_obs = A[:, 0]
#     B_obs = B[:, 0]
#
#     for run in pyro.plate("runs", runs):
#
#         mean_A = mu_A + torch.sum(brA[run] * A[:, 2 + run])
#
#         run_A = pyro.sample("runA_{0}".format(run), dist.Normal(mean_A, sigma_run_A))
#
#         mean_B = mu_B + run_A*bB + torch.sum(brB[run] * B[:, 2 + run])
#
#         run_B = pyro.sample("runB_{0}".format(run), dist.Normal(mean_B, sigma_run_B))
#
#         for feature in pyro.plate("features_{0}".format(run), features):
#             featureA_mean = run_A + torch.sum(bfA[feature] * A[:, 2 + run + feature])
#
#             pyro.sample("obsA_{0}_{1}".format(run, feature),
#                         dist.Normal(featureA_mean, sigma_feature),
#                         obs=A_obs[run+feature])
#
#             featureB_mean = run_B + torch.sum(bfB[feature] * B[:, 2 + run + feature])
#
#             pyro.sample("obsB_{0}_{1}".format(run, feature), dist.Normal(featureB_mean, sigma_feature),
#                         obs=B_obs[run+feature])
#
#         # mnar = 1/ (1 + torch.exp(-beta0 + (beta1 * mean_A)))
#         #
#         # print(mean_A, dist.Bernoulli(probs=mnar).sample())
#         # missing = pyro.sample("missingA_{0}".format(obs),
#         #                       dist.Bernoulli(probs=mnar),
#         #                       obs=A[obs, 1])
#         # int_mu = pyro.param("int_muA_{0}".format(obs), mean_A)
#         # int_sigma = pyro.param("int_sigmaA_{0}".format(obs), torch.tensor(1.),
#         #                        constraint=constraints.positive)
#         # Aimp = pyro.sample(
#         #     "Aimp_{0}".format(obs),
#         #     dist.Normal(mean_A, sigma_A).mask(False)
#         # )
#         #
#         # A_obs = torch.where((A[obs, 1] == 1), Aimp, A_obs)
#         # with poutine.mask(mask=(A[obs, 1] == 1)):
#
# def two_nodes_guide(A, B, runs, features):
#
#     beta0 = 6.5
#     beta1 = .4
#
#     ## A
#     muA_loc = pyro.param("mu_locA", torch.tensor(16.))
#     muA_scale = pyro.param("mu_scaleA", torch.tensor(5.),
#                           constraint=constraints.positive)
#     mu_A = pyro.sample("mu_A", dist.Normal(muA_loc, muA_scale))
#
#     sigma_run_loc = pyro.param("sigma_run_loc", torch.tensor(1.),
#                             constraint=constraints.positive)
#     sigma_run_scale = pyro.param("sigma_run_scale", torch.tensor(.01),
#                              constraint=constraints.positive)
#     sigma_run_A = pyro.sample("sigma_run_A", dist.Normal(sigma_run_loc, sigma_run_scale))
#
#     sigma_run_locB = pyro.param("sigma_run_locB", torch.tensor(1.),
#                             constraint=constraints.positive)
#     sigma_run_scaleB = pyro.param("sigma_run_scaleB", torch.tensor(.01),
#                              constraint=constraints.positive)
#     sigma_run_B = pyro.sample("sigma_run_B", dist.Normal(sigma_run_locB, sigma_run_scaleB))
#
#     br_locA = pyro.param("br_locA", torch.randn(runs))
#     br_scaleA = pyro.param("br_scaleA", torch.ones(runs), constraint=constraints.positive)
#
#     bf_locA = pyro.param("bf_locA", torch.randn(features))
#     bf_scaleA = pyro.param("bf_scaleA", torch.ones(features), constraint=constraints.positive)
#
#     ## B
#     bB_loc = pyro.param("bB_loc", torch.tensor(0.))
#     bB_scale = pyro.param("bB_scale", torch.tensor(0.1),
#                           constraint=constraints.positive)
#     bB = pyro.sample("bB", dist.Normal(bB_loc, bB_scale))
#
#     muB_loc = pyro.param("mu_locB", torch.tensor(20.))
#     muB_scale = pyro.param("mu_scaleB", torch.tensor(5.),
#                           constraint=constraints.positive)
#     mu_B = pyro.sample("mu_B", dist.Normal(muB_loc, muB_scale))
#
#     sigma_feature_loc = pyro.param("sigma_feature_loc", torch.tensor(1.))
#     sigma_feature_scale = pyro.param("sigma_feature_scale", torch.tensor(.1),
#                              constraint=constraints.positive)
#
#     sigma_feature = pyro.sample("sigma_feature", dist.Normal(sigma_feature_loc, sigma_feature_scale))
#
#     br_locB = pyro.param("br_locB", torch.randn(runs))
#     br_scaleB = pyro.param("br_scaleB", torch.ones(runs), constraint=constraints.positive)
#
#     bf_locB = pyro.param("bf_locB", torch.randn(features))
#     bf_scaleB = pyro.param("bf_scaleB", torch.ones(features), constraint=constraints.positive)
#
#     with pyro.plate("runs_param", runs):
#         brA = pyro.sample("bR_A", dist.Normal(br_locA, br_scaleA))
#         brB = pyro.sample("bR_B", dist.Normal(br_locB, br_scaleB))
#     with pyro.plate("features_param", features):
#         pyro.sample("bF_A", dist.Normal(bf_locA, bf_scaleA))
#         pyro.sample("bF_B", dist.Normal(bf_locB, bf_scaleB))
#
#     for run in pyro.plate("runs", runs):
#         mean_A = mu_A + torch.sum(brA[run] * A[:, 2+run])
#
#         run_A = pyro.sample("runA_{0}".format(run), dist.Normal(mean_A, sigma_run_A))
#
#         mean_B = mu_B + run_A*bB + torch.sum(brB[run] * B[:, 2 + run])
#
#         pyro.sample("runB_{0}".format(run), dist.Normal(mean_B, sigma_run_B))

