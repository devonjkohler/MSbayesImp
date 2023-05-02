
import numpyro
from jax import numpy as jnp
from jax import random
import arviz as az

from numpyro.infer import MCMC, NUTS


import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix

from simulation_code import DataSimulator

import pickle

def two_nodes_run_level(data, missing, network, runs=5, features=5, learn_priors_from_data=True,
                        priors=None):

    ## Initialize model
    nodes = list(nx.topological_sort(network))

    # beta0 = 6.5
    # beta1 = .4
    beta0 = numpyro.sample("beta0", numpyro.distributions.Normal(10., 5.))
    beta1 = numpyro.sample("beta1", numpyro.distributions.Normal(1., .25))
    mar = numpyro.sample("mar", numpyro.distributions.LogNormal(-3, .1))

    run_mu_list = list()
    feature_mu_list = list()
    sigma_list = list()

    if learn_priors_from_data:

        for i in range(len(data)):

            run_mu_list.append(numpyro.sample("mu_{}".format(i),
                                              numpyro.distributions.Normal(priors["run_effect_{}".format(i)], 10.)))

            ## Feature means
            feature_mu_list.append(numpyro.sample("bF_{}".format(i),
                                                  # numpyro.distributions.Normal(0., 2.).expand([features])))
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
        run_mu_temp = run_mu_list[i][data[i][:, 0].astype(int)]
        feature_mu_temp = feature_mu_list[i][data[i][:, 1].astype(int)]
        sigma_temp = sigma_list[i]

        temp_missing = missing[i]

        mnar_missing = 1 / (1 + jnp.exp(-beta0 + (beta1 * (run_mu_temp + feature_mu_temp)) - (.5*beta1*sigma_temp)))
        mnar_not_missing = 1 / (1 + jnp.exp(-beta0 + (beta1 * (run_mu_temp + feature_mu_temp))))
        mnar = jnp.where(temp_missing == 1, mnar_missing, mnar_not_missing)

        missing_prob = mar + ((1 - mar) * mnar)

        numpyro.distributions.constraints.positive(numpyro.sample("missing_{}".format(i),
                              numpyro.distributions.Bernoulli(probs=missing_prob),
                              obs=temp_missing))
        adjustment = mnar/(mar+mnar)*(.5 * beta1 * sigma_temp)#

        imp = numpyro.sample(
            "imp_{}".format(i), numpyro.distributions.Normal(
                run_mu_temp[temp_missing] + feature_mu_temp[temp_missing] - adjustment[temp_missing],
                sigma_temp).mask(False)
        )

        obs = data[i][:, 2]
        observed = jnp.asarray(obs).at[temp_missing].set(imp)
        mean = jnp.where(temp_missing==1,
                         run_mu_temp + feature_mu_temp - adjustment,
                         run_mu_temp + feature_mu_temp)

        numpyro.sample("obs_{}".format(i), numpyro.distributions.Normal(mean, sigma_temp), obs=observed)


# squared exponential kernel with diagonal noise term
def kernel(X, alpha, std):
    # dist = lambda p1, p2: np.sqrt(((p1 - p2) ** 2).sum())
    # dm = np.asarray([[dist(p1, p2) for p2 in X] for p1 in X])
    dm = np.cov(X)
    k = alpha * jnp.exp(-(dm/std))
    return k

def causal_model(posterior_samples, obs_samples):

    var = numpyro.sample("kernel_var", numpyro.distributions.LogNormal(0.0, 10.0))
    noise = numpyro.sample("kernel_noise", numpyro.distributions.LogNormal(0.0, 10.0))

    K = kernel(obs_samples, var, noise)
    # K = np.cov(obs_samples)
    posterior_samples = np.array(posterior_samples)

    for i in range(len(posterior_samples[0])):

        obs = posterior_samples[:, i]
        missing = (obs == 0)
        means = jnp.array([5 for _ in range(len(posterior_samples[:, 0]))])
        ## Impute missing values with some mean
        if sum(missing) > 0:
            imp = numpyro.sample(
                "run_imp_{}".format(i), numpyro.distributions.Normal(
                    means[missing],
                    2.).mask(False)
            )

            ## Update obs with imp
            obs = jnp.asarray(obs).at[missing].set(imp)

        ## Use multivariate normal with covariance matrix and obs
        test = numpyro.sample("obs_{}".format(i),
                       numpyro.distributions.MultivariateNormal(loc=means, covariance_matrix=K),
                       obs=jnp.asarray(obs))


class CausalModel:

    def __init__(self, network):
        self.causal_network = network
        self.priors = None
        pass

    def format_data(self, data):

        input_data = dict()
        model_params = dict()

        formatted_data = data.loc[:, ["Protein", "Run", "Feature", "Intensity", "Missing"]]
        formatted_data.loc[:, "Dummy_Run"] = formatted_data.loc[:, "Run"].astype(str)
        formatted_data.loc[:, "Dummy_Feature"] = formatted_data.loc[:, "Feature"].astype(str)

        for i in formatted_data["Protein"].unique():
            temp_data = formatted_data.loc[formatted_data["Protein"] == i]
            temp_data = pd.get_dummies(temp_data, drop_first=False)
            temp_data.loc[:, "Missing"] = np.where(np.isnan(temp_data["Intensity"]), 1., 0.)
            n_runs = len(temp_data.columns[temp_data.columns.str.contains("Dummy_Run")])
            n_feat = len(temp_data.columns[temp_data.columns.str.contains("Dummy_Feature")])
            model_params[i] = {"Runs" : n_runs,
                               "Features" : n_feat}
            temp_data = jnp.asarray(temp_data.drop(columns="Protein").values)
            temp_data = np.nan_to_num(temp_data, nan=0.)
            input_data[i] = temp_data

        return input_data, model_params

    def id_no_impute(self, data, features=10, runs=20):

        ## Get runs with all missing values
        missing_runs = data["Intensity"].isnull().groupby(
            [data["Protein"], data["Run"]]).sum().reset_index()

        total_obs = parameters["Features"]
        missing_runs = missing_runs.loc[missing_runs["Intensity"] == total_obs]


        ## Get features with at least 2 obs
        missing_features = data["Intensity"].isnull().groupby(
            [data["Protein"], data["Feature"]]).sum().reset_index()
        total_runs = parameters["Replicate"]
        missing_features = missing_features.loc[missing_features["Intensity"] > total_runs-2]

        data.loc[:, "Impute"] = 1
        impute_data=data

        if len(missing_runs) > 0:
            missing_runs.loc[:, "Impute"] = 0
            impute_data = pd.merge(impute_data, missing_runs.loc[:, ["Protein", "Run", "Impute"]], how="left",
                                  on=["Protein", "Run"])
            impute_data.loc[:, "Impute"] = np.where(np.isnan(impute_data.loc[:, "Impute_y"]),
                                                    impute_data.loc[:, "Impute_x"],
                                                    impute_data.loc[:, "Impute_y"])
            impute_data = impute_data.drop(columns=["Impute_x", "Impute_y"])

        if len(missing_features) > 0:
            missing_features.loc[:, "Impute"] = 0
            impute_data = pd.merge(impute_data, missing_features.loc[:, ["Protein", "Feature", "Impute"]], how="left",
                                  on=["Protein", "Feature"])
            impute_data.loc[:, "Impute"] = np.where(np.isnan(impute_data.loc[:, "Impute_y"]),
                                                    impute_data.loc[:, "Impute_x"],
                                                    impute_data.loc[:, "Impute_y"])
            impute_data = impute_data.drop(columns=["Impute_x", "Impute_y"])

        return impute_data

    def get_priors(self, data, network):

        nodes = list(nx.topological_sort(network))
        priors = dict()

        for i in nodes:
            runs = len(np.unique(data[i][:, 0]))
            features = len(np.unique(data[i][:, 1]))

            ## Overall mean
            overall_mean = data[i][:, 2][data[i][:, 2] != 0].mean()

            run_effect = list()
            run_std = list()
            run_effect = list()
            run_std = list()

            for r in range(runs):
                # run_effect.append(data[i][:, 2][(data[i][:, 0] == r) & (data[i][:, 2] != 0)].mean() - overall_mean)
                # run_std.append(data[i][:, 2][(data[i][:, 0] == r) & (data[i][:, 2] != 0)].std())
                run_effect.append(0.)
                run_std.append(1.)


            run_effect = jnp.array(run_effect)
            run_effect = np.nan_to_num(run_effect, nan=0.)
            run_std = jnp.array(run_std)
            run_std = np.nan_to_num(run_std, nan=1.)

            feature_effect = list()
            feature_std = list()

            for f in range(features):
                ## TODO: probably remove the run effect here as well
                # feature_effect.append(data[i][:, 2][(data[i][:, 1] == f) & (data[i][:, 2] != 0)].mean() - overall_mean)
                # feature_std.append(data[i][:, 2][(data[i][:, 1] == f) & (data[i][:, 2] != 0)].std())
                feature_effect.append(0.)
                feature_std.append(1.)

            feature_effect = jnp.array(feature_effect)
            feature_effect = np.nan_to_num(feature_effect, nan=0.)
            feature_std = jnp.array(feature_std)
            feature_std = np.nan_to_num(feature_std, nan=1.)

            priors["mean_{}".format(i)] = overall_mean
            priors["run_effect_{}".format(i)] = run_effect
            priors["run_std_{}".format(i)] = run_std
            priors["feature_effect_{}".format(i)] = feature_effect
            priors["feature_std_{}".format(i)] = feature_std

        self.priors = priors


    def train(self, data, num_iters, initial_lr, gamma):

        # data = self.id_no_impute(data)

        format_data, params = self.format_data(data)

        missing = list()
        for i in range(len(format_data)):
            missing.append(format_data[i][:, 3] == 1.)

        self.get_priors(format_data, self.causal_network)

        mcmc = MCMC(NUTS(two_nodes_run_level), num_warmup=1000, num_samples=1000)
        mcmc.run(random.PRNGKey(69), format_data, missing, self.causal_network, priors=self.priors)
        samples = mcmc.get_samples()

        nodes = list(nx.topological_sort(self.causal_network))

        ## Sample run values from posterior
        missing = list()
        posterior_samples = list()
        for p in nodes:
            temp_missing = list()
            temp_posterior_samples = list()
            for i in range(100):
                temp_samples = samples["mu_{}".format(p)][:, i]
                mean = temp_samples.mean()
                std = temp_samples.std()
                if std > 3:
                    temp_missing.append(True)
                    temp_posterior_samples.append(0)
                else:
                    temp_missing.append(False)
                    temp_posterior_samples.append(np.random.normal(mean, std))
            missing.append(jnp.array(temp_missing))
            posterior_samples.append(jnp.array(temp_posterior_samples))

        obs_samples = np.array(posterior_samples)
        obs_samples = obs_samples.T[~(obs_samples.T == 0).any(axis=1)].T

        run_mcmc = MCMC(NUTS(causal_model), num_warmup=500, num_samples=500)
        run_mcmc.run(random.PRNGKey(69), posterior_samples, obs_samples)

        # print(mcmc.print_summary())
        # param_names = [x for x in list(samples.keys()) if "bC" in x]
        # for i in param_names:
        #     print(i)
        #     print(samples[i].mean())

        dist = lambda p1, p2: np.sqrt(((p1 - p2) ** 2).sum())
        dm = np.asarray([[dist(p1, p2) for p2 in obs_samples] for p1 in obs_samples])

        with open(r"data/dm.pickle", "wb") as output_file:
            pickle.dump(dm, output_file)

        idata = az.from_numpyro(mcmc)
        with open(r"data/az_mcmc_gp_two.pickle", "wb") as output_file:
            pickle.dump(idata, output_file)

        idata = az.from_numpyro(run_mcmc)
        with open(r"data/az_run_mcmc_gp_two.pickle", "wb") as output_file:
            pickle.dump(idata, output_file)

def main():
    with open(r"data/simulated_data_ten.pickle", "rb") as input_file:
        simulator = pickle.load(input_file)

    model = CausalModel(simulator.network)
    model.train(simulator.data, 1000, .01, .01)

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


## New old code
    # ## Initialize model
    # nodes = list(nx.topological_sort(network))
    #
    # beta0 = numpyro.sample("beta0", numpyro.distributions.Normal(10., 1.))
    # beta1 = numpyro.sample("beta1", numpyro.distributions.Normal(1., .1))
    # mar = numpyro.sample("mar", numpyro.distributions.LogNormal(-3, .2))
    # #
    # beta0_list = list()
    # causal_effect_list = list()
    # run_mu_list = list()
    # feature_mu_list = list()
    # sigma_list = list()
    # #
    # # if learn_priors_from_data:
    # #
    # #     for i in nodes:
    # #
    # #         ## Intercept(-tion)
    # #         beta0_list.append(numpyro.sample("mu_{}".format(i),
    # #                                           numpyro.distributions.Normal(
    # #                                               10.,#priors["mean_{}".format(i)],
    # #                                               10.)))
    # #
    # #         ## Causal effect
    # #         causal_effect = jnp.asarray([0])
    # #         in_edges = list(network.in_edges(i))
    # #         if len(in_edges) > 0:
    # #             for e in in_edges:
    # #                 causal_param = numpyro.sample("bC_{0}_{1}".format(i, e[0]),
    # #                                               numpyro.distributions.Normal(0., .3))
    # #                 causal_effect += causal_param * (beta0_list[e[0]] + run_mu_list[e[0]])
    # #         causal_effect_list.append(causal_effect)
    # #
    # #         ## Run effect
    # #         run_mu_list.append(numpyro.sample("bR_{}".format(i),
    # #                                           numpyro.distributions.Normal(
    # #                                               priors["run_effect_{}".format(i)],
    # #                                               1.)))
    # #
    # #         ## Feature means
    # #         feature_mu_list.append(numpyro.sample("bF_{}".format(i),
    # #                                               numpyro.distributions.Normal(priors["feature_effect_{}".format(i)],
    # #                                                                            1.)))
    # #
    # #         ## Error
    # #         sigma_list.append(numpyro.sample("error_{}".format(i), numpyro.distributions.Exponential(1.)))
    # #
    # # else:
    # #     for i in range(len(data)):
    # #         run_mu_list.append(numpyro.sample("mu_{}".format(i), numpyro.distributions.Uniform(10, 20).expand([runs])))
    # #
    # #         ## Feature means
    # #         feature_mu_list.append(numpyro.sample("bF_{}".format(i),
    # #                                               numpyro.distributions.Normal(0., 2.).expand([features])))
    # #
    # #         sigma_list.append(numpyro.sample("error_{}".format(i), numpyro.distributions.Uniform(0., 2.)))
    #
    # observation_tracker = dict()
    #
    # for i in nodes:
    #     beta0_temp = numpyro.sample("mu_{}".format(i), numpyro.distributions.Normal(10., 10.))
    #     beta0_list.append(beta0_temp)
    #
    #     # causal_effect = jnp.asarray([0])
    #     # in_edges = list(network.in_edges(i))
    #     # if len(in_edges) > 0:
    #     #     for e in in_edges:
    #     #         causal_param = numpyro.sample("bC_{0}_{1}".format(i, e[0]),
    #     #                                       numpyro.distributions.Normal(0., .3))
    #     #         causal_effect += causal_param * (beta0_list[e[0]] + run_mu_list[e[0]])
    #     # else:
    #     causal_effect = jnp.asarray([0 for _ in range(len(np.unique(data[i][:, 0].astype(int))))])
    #
    #     # run_mu_temp = run_mu_list[i][data[i][:, 0].astype(int)]
    #     # feature_mu_temp = feature_mu_list[i][data[i][:, 1].astype(int)]
    #     sigma_temp = numpyro.sample("error_{}".format(i), numpyro.distributions.Exponential(1.))
    #     temp_missing = missing[i]
    #
    #     run_mean = numpyro.sample(
    #         "run_{}".format(i), numpyro.distributions.Normal(
    #             beta0_temp + causal_effect,
    #             sigma_temp)
    #     )
    #     run_mu_list.append(run_mean)
    #
    #     run_mean_temp = run_mean[data[i][:, 0].astype(int)]
    #     feature_effect = numpyro.sample("bF_{}".format(i),
    #                                     numpyro.distributions.Normal(priors["feature_effect_{}".format(i)], 1.))
    #     feature_effect = feature_effect[data[i][:, 1].astype(int)]
    #
    #     mnar_missing = 1 / (1 + jnp.exp(-beta0 + (beta1 * ((run_mean_temp + feature_effect) - (.5*beta1*sigma_temp)))))
    #     mnar_not_missing = 1 / (1 + jnp.exp(-beta0 + (beta1 * (run_mean_temp + feature_effect))))
    #     mnar = jnp.where(temp_missing == 1, mnar_missing, mnar_not_missing)
    #
    #     missing_prob = mar + ((1 - mar) * mnar)
    #
    #     numpyro.distributions.constraints.positive(numpyro.sample("missing_{}".format(i),
    #                           numpyro.distributions.Bernoulli(probs=missing_prob),
    #                           obs=temp_missing))
    #
    #     imp_mean = run_mean_temp[temp_missing] + feature_effect[temp_missing] - (.5 * beta1 * sigma_temp)
    #
    #     imp = numpyro.sample(
    #         "imp_{}".format(i), numpyro.distributions.Normal(
    #             imp_mean,
    #             sigma_temp).mask(False)
    #     )
    #
    #     obs = data[i][:, 2]
    #     observed = jnp.asarray(obs).at[temp_missing].set(imp)
    #     final_mean = jnp.where(temp_missing == 1, (run_mean_temp + feature_effect) - (.5 * beta1 * sigma_temp),
    #                            (run_mean_temp + feature_effect))
    #
    #     observation_tracker[i] = numpyro.sample("obs_{}".format(i),
    #                                            numpyro.distributions.Normal(final_mean, sigma_temp),
    #                                            obs=observed)