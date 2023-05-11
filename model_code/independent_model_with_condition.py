
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
from numpyro.distributions import constraints
numpyro.set_platform('cpu')
# numpyro.set_host_device_count(4)

import jax
from jax import numpy as jnp
from jax import random
# import az.from_numpyro as az_numpy
# import arviz as az

import time
import pickle
from operator import attrgetter

import pandas as pd
import numpy as np

# import simulation_code.Data

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
print(jax.local_device_count())

class beta0TruncatedNormal(numpyro.distributions.Normal):
    support = constraints.interval(0., 20.)
    def sample(self, key, sample_shape=()):
        return numpyro.distributions.TruncatedNormal(self.loc, self.scale, low=0.
                                                     ).sample(key, sample_shape=sample_shape)
    def log_prob(self, value):
        return numpyro.distributions.TruncatedNormal(self.loc, self.scale, low=0.).log_prob(value)

class beta1TruncatedNormal(numpyro.distributions.Normal):
    support = constraints.interval(.01, 1.)
    def sample(self, key, sample_shape=()):
        return numpyro.distributions.TruncatedNormal(self.loc, self.scale, low=0.01
                                                     ).sample(key, sample_shape=sample_shape)
    def log_prob(self, value):
        return numpyro.distributions.TruncatedNormal(self.loc, self.scale, low=0.01).log_prob(value)

class marTruncatedNormal(numpyro.distributions.Normal):
    support = constraints.interval(.001, .1)
    def sample(self, key, sample_shape=()):
        return numpyro.distributions.TruncatedNormal(self.loc, self.scale, low=0.001
                                                     ).sample(key, sample_shape=sample_shape)
    def log_prob(self, value):
        return numpyro.distributions.TruncatedNormal(self.loc, self.scale, low=0.001).log_prob(value)

class runTruncatedNormal(numpyro.distributions.Normal):
    support = constraints.interval(1., 40.)
    def sample(self, key, sample_shape=()):
        return numpyro.distributions.TruncatedNormal(self.loc, self.scale, low=1.
                                                     ).sample(key, sample_shape=sample_shape)
    def log_prob(self, value):
        return numpyro.distributions.TruncatedNormal(self.loc, self.scale, low=1.).log_prob(value)

def feature_level_model(data, missing, priors, condition_to_run_mapper):

    ## Initialize experiment wide params
    beta0 = numpyro.sample("beta0", beta0TruncatedNormal(4., 1.))
    beta1 = numpyro.sample("beta1", beta1TruncatedNormal(.5, .25))
    # mar = numpyro.sample("mar", marTruncatedLogNormal(-3.5, .0001))
    mar = numpyro.sample("mar", marTruncatedNormal(.03, .0001))

    ## Initialize model variables
    condition_mu_list = numpyro.sample("bC", runTruncatedNormal(priors["cond_effect"], 4.))
    condition_mu = condition_mu_list[condition_to_run_mapper[:, 0].astype(int)]

    run_mu_list = numpyro.sample("bR", runTruncatedNormal(condition_mu, 4.))
    feature_mu_list = numpyro.sample("bF", numpyro.distributions.Normal(priors["feature_effect"], 1.))
    sigma = numpyro.sample("error", numpyro.distributions.Exponential(1.).expand([len(data)]))

    ## Get model param for each obs
    run_mu = run_mu_list[data[:, 1].astype(int)]
    feature_mu = feature_mu_list[data[:, 2].astype(int)]

    # sigma = sigma_list[data[:, 1].astype(int)]

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
        "imp", runTruncatedNormal(
            imp_means[missing==1],
            sigma[missing==1]).mask(False)
    )

    ## Add imputed missing values to observations
    obs = data[:, 3]
    observed = jnp.asarray(obs).at[missing==1].set(imp)

    ## Sample with obs
    mean = jnp.where(missing==1,
                     run_mu + feature_mu - adjustment,
                     run_mu + feature_mu)

    numpyro.sample("obs", runTruncatedNormal(mean, sigma), obs=observed)

class IndependentModel:

    def __init__(self):

        # TODO: Figure out which of these are really necessary
        self.priors = None
        self.original_data = None
        self.no_impute_df = None
        self.flat_input = None
        self.lookup_table = None
        self.summary_stats = None
        self.mcmc_samples = None
        self.results_df = None

    def format_data(self, data):

        formatted_data = data.loc[:, ["Protein", "Condition", "Run", "Feature", "Intensity", "Missing"]]
        for col in ["Protein", "Condition", "Run", "Feature"]:
            if formatted_data[col].dtype == 'O':
                formatted_data.loc[:, "{0}_original".format(col)] = formatted_data.loc[:, col]
                formatted_data.loc[:, col] = formatted_data.loc[:, col].astype("category").cat.codes
        self.original_data = formatted_data

        # Drop a pandas column if it contains a string
        formatted_data = formatted_data[formatted_data.columns.drop(list(formatted_data.filter(regex='original')))]

        ## TODO: This is written for the simulated data.. not needed for real data...
        # formatted_data.loc[:, "Condition_run"] = formatted_data.loc[:, "Condition"].astype(str) + "_" + \
        #                                          formatted_data.loc[:, "Run"].astype(str)
        # formatted_data.loc[:, "Run"] = formatted_data.loc[:, "Condition_run"].astype("category").cat.codes
        # formatted_data = formatted_data.drop(columns=["Condition", "Condition_run"])
        # formatted_data = formatted_data.drop(columns=["Condition"])

        formatted_data.loc[:, "Missing"] = np.where(np.isnan(formatted_data["Intensity"]), 1., 0.)
        formatted_data = formatted_data.sort_values(by=["Protein", "Feature", "Condition", "Run"])

        formatted_data = formatted_data.fillna(0.)

        # Identify values that cannot be imputed
        formatted_data = self.id_no_impute(formatted_data)
        self.no_impute_df = formatted_data[formatted_data["can_imp"] == 0]

        formatted_data = formatted_data[formatted_data["can_imp"] == 1]
        formatted_data = formatted_data.drop(columns=["can_imp"])

        return formatted_data.values

    def id_no_impute(self, data):

        # Get runs with all missing values
        missing_runs = data.groupby([data["Protein"], data["Run"]])["Missing"].sum().reset_index()

        total_obs = data.groupby(["Protein", "Run"])["Feature"].count().values
        missing_runs = missing_runs.loc[missing_runs["Missing"] == total_obs]

        # Get features with at least 2 obs
        missing_features = data.groupby([data["Protein"], data["Feature"]])["Missing"].sum().reset_index()
        total_runs = data.groupby(["Protein", "Feature"])["Run"].count().values
        missing_features = missing_features.loc[missing_features["Missing"] > total_runs-2]

        data.loc[:, "can_imp"] = 1

        if len(missing_runs) > 0:
            missing_runs.loc[:, "can_imp"] = 0
            data = pd.merge(data, missing_runs.loc[:, ["Protein", "Run", "can_imp"]], how="left",
                                  on=["Protein", "Run"])
            data.loc[:, "can_imp"] = np.where(np.isnan(data.loc[:, "can_imp_y"]),
                                                    data.loc[:, "can_imp_x"],
                                                    data.loc[:, "can_imp_y"])
            data = data.drop(columns=["can_imp_x", "can_imp_y"])

        if len(missing_features) > 0:
            missing_features.loc[:, "can_imp"] = 0
            data = pd.merge(data, missing_features.loc[:, ["Protein", "Feature", "Impute"]], how="left",
                                  on=["Protein", "Feature"])
            data.loc[:, "can_imp"] = np.where(np.isnan(data.loc[:, "can_imp_y"]),
                                                    data.loc[:, "can_imp_x"],
                                                    data.loc[:, "can_imp_y"])
            data = data.drop(columns=["can_imp_x", "can_imp_y"])

        return data

    def flatten_input(self, data):

        data = pd.DataFrame(data, columns=["Protein", "Condition", "Run", "Feature", "Intensity", "Missing"])

        # Format input data to be ready for flatten
        data.loc[:, "list_index"] = np.arange(len(data))
        data.loc[:, "Protein_condition"] = data.loc[:, "Protein"].astype(str) + "_" + data.loc[:, "Condition"].astype(str)
        data.loc[:, "Protein_run"] = data.loc[:, "Protein"].astype(str) + "_" + data.loc[:, "Run"].astype(str)
        data.loc[:, "Protein_feature"] = data.loc[:, "Protein"].astype(str) + "_" + data.loc[:, "Feature"].astype(str)

        # TODO: This is sorta gross but cat.codes doesn't create these in order
        data = pd.merge(data, pd.DataFrame({
            "Protein_condition": data.loc[:, "Protein_condition"].unique(),
            "Protein_condition_idx": np.arange(len(data.loc[:, "Protein_condition"].unique()))}),
                 on="Protein_condition", how="left")
        data = pd.merge(data, pd.DataFrame({"Protein_run": data.loc[:, "Protein_run"].unique(),
                      "Protein_run_idx" : np.arange(len(data.loc[:, "Protein_run"].unique()))}),
                 on="Protein_run", how="left")
        data = pd.merge(data, pd.DataFrame({
            "Protein_feature": data.loc[:, "Protein_feature"].unique(),
            "Protein_feature_idx": np.arange(len(data.loc[:, "Protein_feature"].unique()))}),
                 on="Protein_feature", how="left")

        # Return data for model and interpretation
        flatten_data = data.loc[:, ["Protein_condition_idx", "Protein_run_idx",
                                    "Protein_feature_idx", "Intensity"]].values
        self.flat_input = flatten_data

        condition_to_run_mapper = data.loc[:, ["Protein_condition_idx", "Run"]
                                  ].drop_duplicates(ignore_index=True).values
        self.condition_to_run_mapper = condition_to_run_mapper
        self.lookup_table = data

    def get_priors(self, data):

        # Initialize collection structures
        cond_priors = list()
        run_priors = list()
        run_priors_std = list()
        feature_priors = list()

        proteins = np.unique(data[:, 0])

        for i in range(len(proteins)):

            temp_data = data[data[:, 0] == proteins[i]]

            conditions = np.unique(temp_data[:, 1])
            runs = np.unique(temp_data[:, 2])
            features = np.unique(temp_data[:, 3])

            # Overall mean
            overall_mean = temp_data[:, 4][temp_data[:, 4] != 0].mean()

            # Calculate condition priors
            cond_effect = list()

            for c in conditions:
                cond_effect.append(temp_data[:, 4][(temp_data[:, 1] == c) & (temp_data[:, 4] != 0)].mean())

            cond_effect = np.array(cond_effect)
            cond_effect = np.nan_to_num(cond_effect, nan=0.)
            cond_priors.append(cond_effect)

            # Calculate run priors
            run_effect = list()
            run_std = list()

            for r in runs:
                run_effect.append(temp_data[:, 4][(temp_data[:, 2] == r) & (temp_data[:, 4] != 0)].mean())
                run_std.append(temp_data[:, 4][(temp_data[:, 2] == r) & (temp_data[:, 4] != 0)].var())

            run_effect = np.array(run_effect)
            run_effect = np.nan_to_num(run_effect, nan=0.)
            run_priors.append(run_effect)

            run_std = np.array(run_std)
            run_std = np.nan_to_num(run_std, nan=0.1)
            run_std = np.where(run_std == 0., .75, run_std)
            run_priors_std.append(run_std)

            # Calculate feature priors
            feature_effect = list()
            feature_std = list()

            for f in features:
                feature_effect.append(temp_data[:, 4][(temp_data[:, 3] == f) &
                                                      (temp_data[:, 4] != 0)].mean() - overall_mean)
                feature_std.append(temp_data[:, 4][(temp_data[:, 3] == f) & (temp_data[:, 4] != 0)].std())

            feature_effect = np.array(feature_effect)
            feature_effect = np.nan_to_num(feature_effect, nan=0.)
            feature_priors.append(feature_effect)

        priors = dict()

        cond_priors = np.concatenate(cond_priors)
        run_priors = np.concatenate(run_priors)
        run_priors_std = np.concatenate(run_priors_std)
        feature_priors = np.concatenate(feature_priors)

        priors["cond_effect"] = cond_priors
        priors["run_effect"] = run_priors
        priors["run_std"] = run_priors_std
        priors["feature_effect"] = feature_priors

        self.priors = priors

    def compile_model_stats(self, model, prob=.9):

        """
        By default, numpyro only provides model summary statistics in the `print_summary()` function. This is a pain
        because it cannot be saved to a variable and cannot be read with thousands of parameters learned. This function
        simply runs the underlying code for `print_summary()` and returns the summary statistics.

        :param model: trained mcmc model
        :param prob: the probability mass of samples within the HPDI interval.

        :return: dictionary of summary statistics.
        """

        sites = model._states[model._sample_field]

        if isinstance(sites, dict):
            state_sample_field = attrgetter(model._sample_field)(model._last_state)
            if isinstance(state_sample_field, dict):
                sites = {
                    k: v
                    for k, v in model._states[model._sample_field].items()
                    if k in state_sample_field
                }

        summary_stats = numpyro.diagnostics.summary(sites, prob, group_by_chain=True)
        self.summary_stats = summary_stats

    def train(self,
              data,
              warmup_steps,
              sample_steps,
              save_final_state=False,
              save_folder="",
              load_previous_state=False,
              previous_state=None):

        start = time.time()

        # Format data for training
        format_data = self.format_data(data)
        missing = format_data[:, 5]

        # Get priors
        self.get_priors(format_data)
        self.flatten_input(format_data)

        if load_previous_state:
            with open(previous_state, "rb") as input_file:
                mcmc = pickle.load(input_file)
            mcmc.post_warmup_state = mcmc.last_state
            mcmc.run(random.PRNGKey(69), self.flat_input, missing, self.priors, self.condition_to_run_mapper)
        else:
            mcmc = MCMC(NUTS(feature_level_model), num_warmup=warmup_steps, num_samples=sample_steps, num_chains=1)  #
            mcmc.run(random.PRNGKey(69), self.flat_input, missing, self.priors, self.condition_to_run_mapper)

        finish = time.time()
        keep = finish-start
        print("Time to train: {}".format(keep))

        self.compile_model_stats(mcmc)
        print(self.summary_stats)

        # idata = az.from_numpyro(mcmc)
        # self.az_mcmc_results = idata
        self.mcmc_samples = mcmc.get_samples()
        if save_final_state:
            # mcmc._cache = {}
            with open(r"{0}mcmc_state.pickle".format(save_folder), "wb") as output_file:
                pickle.dump(mcmc, output_file)

    def compile_results(self, sim=False):

        ## Create dataframes to join in values
        condition_join_df = pd.DataFrame({"Protein_condition_idx": np.arange(self.mcmc_samples["bC"].shape[1]),
                                        "mean_condition": self.mcmc_samples["bC"].mean(axis=0),
                                        "std_condition": self.mcmc_samples["bC"].std(axis=0)})
        feature_join_df = pd.DataFrame({"Protein_feature_idx": np.arange(self.mcmc_samples["bF"].shape[1]),
                                        "mean_feature": self.mcmc_samples["bF"].mean(axis=0),
                                        "std_feature": self.mcmc_samples["bF"].std(axis=0)})
        run_join_df = pd.DataFrame({"Protein_run_idx": np.arange(self.mcmc_samples["bR"].shape[1]),
                                    "mean_run": self.mcmc_samples["bR"].mean(axis=0),
                                    "std_run": self.mcmc_samples["bR"].std(axis=0)})
        lookup_table = pd.merge(self.lookup_table, run_join_df, on="Protein_run_idx", how="left")
        lookup_table = pd.merge(lookup_table, feature_join_df, on="Protein_feature_idx", how="left")
        lookup_table = pd.merge(lookup_table, condition_join_df, on="Protein_condition_idx", how="left")

        ## recover missing values
        lookup_table.loc[:, "imputation_mean"] = np.nan
        lookup_table.loc[:, "imputation_std"] = np.nan
        lookup_table.loc[lookup_table["Intensity"] == 0, "imputation_mean"] = self.mcmc_samples["imp"].mean(axis=0)
        lookup_table.loc[lookup_table["Intensity"] == 0, "imputation_std"] = self.mcmc_samples["imp"].std(axis=0)

        ## Add back in runs that could not be imputed
        lookup_table = pd.concat([lookup_table, self.no_impute_df.drop(columns=["can_imp"])])


        if not sim:
            lookup_table = pd.merge(lookup_table, self.original_data.loc[:, ["Protein", "Condition", "Run", "Feature",
                                                                             "Protein_original", "Condition_original",
                                                                             "Run_original", "Feature_original"]],
                                    on=["Protein", "Condition", "Run", "Feature"], how="left")

        self.results_df = lookup_table

def main():
    # with open(r"data/simulated_data_200.pickle", "rb") as input_file:
    #     simulator = pickle.load(input_file)
    # input_data = simulator.data
    # input_data = pd.read_csv(r"../data/simulated_data_200.csv")
    # save_folder = r"../model_results/sim200_"

    # save_folder = r"../model_results/Choi2017_cond_"
    save_folder = r"/home/kohler.d/MSbayesImp/model_results/ControlMix_"
    # input_data = pd.read_csv(r"/home/kohler.d/MSbayesImp/model_code/data/Choi2017_model_input.csv")
    input_data = pd.read_csv(r"../data/ControlMix_DDA_input.csv")
    # sample_proteins = np.random.choice(input_data["Protein"].unique(), 19, replace=False)
    # sample_proteins = np.append(sample_proteins, np.array(["O13539"]))
    # sample_proteins = np.array(["O13539", "D6VTK4", "P07275", "P36123", "P53905", "Q03373","P55249"])
    # input_data = input_data.loc[input_data["Protein"].isin(sample_proteins)]

    model = IndependentModel()
    model.train(input_data, 1000, 1000,
                save_final_state=True,
                save_folder=save_folder,
                load_previous_state=False,
                previous_state=r"{0}mcmc_state.pickle".format(save_folder))
    model.compile_results(sim=False)

    with open(r"{0}mcmc_samples.pickle".format(save_folder), "wb") as output_file:
        pickle.dump(model.mcmc_samples, output_file)

    with open(r"{0}results_df.pickle".format(save_folder), "wb") as output_file:
        pickle.dump(model.results_df, output_file)

    with open(r"{0}summary_stats.pickle".format(save_folder), "wb") as output_file:
        pickle.dump(model.summary_stats, output_file)

if __name__ == "__main__":
    main()
