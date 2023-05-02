

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO

import torch
import torch.distributions.constraints as constraints

import pandas as pd
import numpy as np

import pickle

from simulation_code import DataSimulator
from utils import compare_to_gt
from pyro.infer import MCMC, NUTS, Predictive

def regression_model_new(data=None, parameters=None, impute=True):

    ## MAR indicator
    # alpha_q = pyro.sample("alpha_q", dist.Normal(10., .01))
    # beta_q = pyro.sample("beta_q", dist.Normal(100., .1))
    #
    # # mar_prob = pyro.sample("MAR", dist.Beta(alpha, beta))
    # mar = pyro.sample("MAR", dist.Beta(alpha_q, beta_q))
    mar=.05

    beta0_mu = 6.5
    beta1_mu = .4

    beta0 = pyro.sample("beta0", dist.Normal(beta0_mu, torch.tensor(.01)))
    beta1 = pyro.sample("beta1", dist.Normal(beta1_mu, torch.tensor(.0001)))
    # beta0 = 6.5
    # beta1 = .4
    num_proteins = len(data.keys())
    scaler = pyro.param("scaler", torch.tensor(1.),
                        constraint=constraints.positive)
    for protein in pyro.plate("protein", num_proteins):

        protein_name = list(data.keys())[protein]

        protein_data = data[protein_name]
        meta_data = parameters[protein_name]
        num_runs = meta_data["Runs"]
        num_features = meta_data["Features"]
        obs = protein_data.select(1,0)

        mu = pyro.sample("mu_{0}".format(protein_name), dist.Uniform(5. ,40.))
        sigma = pyro.sample("sigma_{0}".format(protein_name),
                            dist.Uniform(0.,100.))

        with pyro.plate("runs_{}".format(protein_name), num_runs):
            br = pyro.sample("bR_{0}".format(protein_name),
                             dist.Normal(torch.zeros(num_runs), torch.ones(num_runs)))
        with pyro.plate("features_{}".format(protein_name), num_features):
            bf = pyro.sample("bF_{0}".format(protein_name),
                             dist.Normal(torch.zeros(num_features), torch.ones(num_features)))

        mean = mu + torch.sum(br  * protein_data[: ,2:2+num_runs], dim=1) + \
               torch.sum(bf * protein_data[: ,2+num_runs:], dim=1)

        with pyro.plate("data_{0}".format(protein_name), len(protein_data[:, 0])):
            mnar = 1/ (1 + torch.exp(-beta0 + (beta1 * mean)))
            missing_prob = mar * ((1 - mar) * mnar)

            missing = pyro.sample("missing_{0}".format(protein_name),
                                  dist.Bernoulli(probs=missing_prob),
                                  obs=protein_data[:, 1])

            if impute:
                int_mu = pyro.sample("int_mu_{0}".format(protein_name), dist.Normal(mean-(mnar*scaler), 1))
                # int_sigma = pyro.sample("int_sigma_{0}".format(protein_name), dist.Normal(1., .01))
                int_sigma = pyro.param("int_sigma_{0}".format(protein_name), torch.tensor(1.),
                        constraint=constraints.positive)
                int_impute = pyro.sample(
                    "int_impute_{0}".format(protein_name),
                    dist.Normal(int_mu, int_sigma).mask(False),
                )

                obs = torch.where((missing==1), int_impute, obs)
                mean = torch.where((missing == 1), mean-(mnar*scaler), mean)

                pyro.sample("obs_{0}".format(protein_name),
                        dist.Normal(mean, sigma), obs=obs)
            else:
                with poutine.mask(mask=(missing == 0)):
                    pyro.sample("obs_{0}".format(protein),
                                dist.Normal(mean, sigma), obs=data[protein, :, 0])

            # with poutine.mask(mask=(missing == 1)):
            #     missing_adj = pyro.sample("adj_{0}".format(protein_name),
            #                               dist.LogNormal(torch.tensor(0.),
            #                                              torch.tensor(1)))
            #     pyro.sample("impute_{0}".format(protein_name),
            #                 dist.Normal(mean - missing_adj, 1.))

def regression_guide_new(data=None, parameters=None, impute=True):

    ## MAR
    # alpha_q_mu = pyro.param("alpha_q_mu", torch.tensor(10.0),
    #                         constraint=constraints.positive)
    # alpha_q_std = pyro.param("alpha_q_std", torch.tensor(1.),
    #                          constraint=constraints.positive)
    # alpha_q = pyro.sample("alpha_q", dist.Normal(alpha_q_mu, alpha_q_std))
    #
    # beta_q_mu = pyro.param("beta_q_mu", torch.tensor(100.0),
    #                        constraint=constraints.positive)
    # beta_q_std = pyro.param("beta_q_std", torch.tensor(5.),
    #                         constraint=constraints.positive)
    # beta_q = pyro.sample("beta_q", dist.Normal(beta_q_mu, beta_q_std))
    #
    # mar = pyro.sample("MAR", dist.Beta(alpha_q, beta_q))
    mar=.05
    # beta0 = 6.5
    # beta1 = .4


    beta0_mu = pyro.param("beta0_mu", torch.tensor(6.5))
    beta1_mu = pyro.param("beta1_mu", torch.tensor(.4))

    beta0 = pyro.sample("beta0", dist.Normal(beta0_mu, torch.tensor(.1)))
    beta1 = pyro.sample("beta1", dist.Normal(beta1_mu, torch.tensor(.01)))
    scaler = pyro.param("scaler", torch.tensor(1.),
                        constraint=constraints.positive)

    num_proteins = len(data.keys())

    for protein in pyro.plate("protein", num_proteins):

        protein_name = list(data.keys())[protein]
        protein_data = data[protein_name]
        meta_data = parameters[protein_name]
        num_runs = meta_data["Runs"]
        num_features = meta_data["Features"]


        ## Sample mean
        mu_loc = pyro.param("mu_loc_{}".format(protein_name), torch.tensor(20.))
        mu_scale = pyro.param("mu_scale_{}".format(protein_name), torch.tensor(2.),
                              constraint=constraints.positive)
        mu = pyro.sample("mu_{}".format(protein_name), dist.Normal(mu_loc, mu_scale))

        ## Sample std
        sigma_loc = pyro.param("sigma_loc_{}".format(protein_name), torch.tensor(1.))
        sigma_scale = pyro.param("sigma_scale_{}".format(protein_name), torch.tensor(.25),
                                 constraint=constraints.positive)
        sigma = pyro.sample("sigma_{}".format(protein_name),
                            dist.LogNormal(sigma_loc, sigma_scale))

        ## Coefficients
        br_loc = pyro.param("br_loc_{0}".format(protein_name), torch.randn(num_runs))
        br_scale = pyro.param("br_scale_{0}".format(protein_name), torch.ones(num_runs),
                              constraint=constraints.positive)

        bf_loc = pyro.param("bf_loc_{0}".format(protein_name), torch.randn(num_features))
        bf_scale = pyro.param("bf_scale_{0}".format(protein_name), torch.ones(num_features),
                              constraint=constraints.positive)

        with pyro.plate("runs_{}".format(protein_name), num_runs):
            br = pyro.sample("bR_{0}".format(protein_name),
                             dist.Normal(br_loc, br_scale))
        with pyro.plate("features_{}".format(protein_name), num_features):
            bf = pyro.sample("bF_{0}".format(protein_name),
                             dist.Normal(bf_loc, bf_scale))

        mean = mu + torch.sum(br  * protein_data[: ,2:2+num_runs], dim=1) + \
               torch.sum(bf * protein_data[: ,2+num_runs:], dim=1)

        if impute:
            with pyro.plate("data_{0}".format(protein_name), len(protein_data[:, 0])):
                mnar = 1/ (1 + torch.exp(-beta0 + (beta1 * mean)))
                missing_prob = mar * ((1 - mar) * mnar)

                # missing = pyro.sample("missing_{0}".format(protein_name),
                #                       dist.Bernoulli(probs=missing_prob),
                #                       infer={'is_auxiliary': True})
                # if sum(missing) > 0:
                int_sigma = pyro.param("int_sigma_{0}".format(protein_name), torch.tensor(1.),
                        constraint=constraints.positive)
                int_mu = pyro.sample("int_mu_{0}".format(protein_name),
                                     dist.Normal(mean - (scaler*mnar), int_sigma))
                # int_sigma = pyro.sample("int_sigma_{0}".format(protein_name), dist.Uniform(0, 1))
                pyro.sample(
                    "int_impute_{0}".format(protein_name),
                    dist.Normal(int_mu, sigma).mask(False),
                )


        #     with poutine.mask(mask=(missing == 1)):
        #         adj_loc = pyro.param("adj_loc_{0}".format(protein_name), torch.zeros(len(protein_data[:, 0])))
        #         adj_std = pyro.param("adj_std_{0}".format(protein_name), torch.ones(len(protein_data[:, 0])),
        #                              constraint=constraints.positive)
        #         missing_adj = pyro.sample("adj_{0}".format(protein_name),
        #                                   dist.LogNormal(adj_loc, adj_std))
        #         pyro.sample("impute_{0}".format(protein_name),
        #                     dist.Normal(mean-missing_adj, sigma))

class IndependentModel:

    def __init__(self):
        self.training_data = None
        self.model_parameters = None
        self.model_results = None

    def format_data(self, data, parameters):
        # formatted_data = data.loc[:, ["Run", "Feature", "Intensity", "Missing", "Impute"]]
        # formatted_data.loc[:, "Run"] = formatted_data.loc[:, "Run"].astype(str)
        # formatted_data.loc[:, "Feature"] = formatted_data.loc[:, "Feature"].astype(str)
        # formatted_data = pd.get_dummies(formatted_data, drop_first=True)
        #
        # formatted_data = formatted_data.values.reshape(parameters["Proteins"],
        #                                                parameters["Replicate"]*parameters["Features"],
        #                                                len(formatted_data.columns))
        #
        # formatted_data = torch.tensor(formatted_data)
        # formatted_data = torch.nan_to_num(formatted_data, 0.)
        # return formatted_data

        input_data = dict()
        model_params = dict()

        formatted_data = data.loc[data["Impute"] == 1][["Protein", "Run", "Feature", "Intensity", "Missing"]]
        formatted_data.loc[:, "Run"] = formatted_data.loc[:, "Run"].astype(str)
        formatted_data.loc[:, "Feature"] = formatted_data.loc[:, "Feature"].astype(str)

        for i in formatted_data["Protein"].unique():
            temp_data = formatted_data.loc[formatted_data["Protein"] == i]
            temp_data = pd.get_dummies(temp_data, drop_first=True)
            n_runs = len(temp_data.columns[temp_data.columns.str.contains("Run")])
            n_feat = len(temp_data.columns[temp_data.columns.str.contains("Feature")])
            model_params[i] = {"Runs" : n_runs,
                               "Features" : n_feat}

            temp_data = torch.tensor(temp_data.drop(columns="Protein").values)
            temp_data = torch.nan_to_num(temp_data, 0.)
            input_data[i] = temp_data

        return input_data, model_params

    def train_model(self, data, parameters, num_iters, initial_lr, gamma):

        self.training_data = data
        input_data, model_params = self.format_data(self.training_data, parameters)
        # torch.autograd.set_detect_anomaly(True)
        # nuts_kernel = NUTS(
        #     regression_model_new
        # )
        # mcmc = MCMC(
        #     nuts_kernel,
        #     num_samples=1000,
        #     warmup_steps=250,
        #     # num_chains=args.num_chains,
        # )
        # mcmc.run(input_data, model_params)

        self.model_parameters = {
            "iterations" : num_iters,
            "learning_rate" : initial_lr,
            "gamma" : gamma
        }

        lrd = gamma ** (1 / num_iters)
        optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})

        svi = SVI(regression_model_new,
                  regression_guide_new,
                  optim,
                  loss=Trace_ELBO())

        pyro.clear_param_store()
        loss_tracker = list()
        current_loss = 0
        loss_tracker.append(svi.step(input_data, model_params))
        i = 0

        while current_loss < np.mean(loss_tracker[max(0, i - 200):i]) or i < 200:
            elbo = svi.step(input_data, model_params)
            loss_tracker.append(elbo)
            current_loss = elbo
            i += 1
            if i % 100 == 0:
                print(str(i) + ": " + str(elbo))

        print("Final loss ({0} iterations): {1}".format(i, current_loss))

        self.model_results = pyro.get_param_store()
        params = dict()
        for key, val in self.model_results.items():
            params[key] = val.detach().numpy()
        with open(r"../models/11_node_model.pickle", "wb") as output_file:
            pickle.dump(params, output_file)

def main():
    with open(r"../data/simulated_data.pickle", "rb") as input_file:
        simulator = pickle.load(input_file)

    model = IndependentModel()

    model.train_model(simulator.data,
                      simulator.sim_parameters,
                      1000, .05, .01)

    proteins = simulator.data["Protein"].unique()
    result_list = list()
    for i in proteins:
        try:
            temp_mean = model.model_results['mu_loc_{0}'.format(i)]
            run_effect = model.model_results['br_loc_{0}'.format(i)]
            run_mean = temp_mean + run_effect
            run_mean = torch.cat((temp_mean.reshape(1), run_mean))
            tmp_results = pd.DataFrame(data=run_mean.reshape(1, len(run_mean)).detach().numpy(),
                                       columns=simulator.data["Run"].unique())
            tmp_results.loc[0, "Protein"] = i
            result_list.append(tmp_results)
        except:
            print("Protein {0} Not found in model".format(i))
    results_df = pd.concat(result_list)
    comparison = compare_to_gt(results_df, simulator.data)
    print(comparison["full"])
    print(comparison["differences"])
    print(comparison["total"])

if __name__ == "__main__":
    main()