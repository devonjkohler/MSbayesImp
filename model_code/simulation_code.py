import random

import pandas as pd
import numpy as np
import pyro.distributions as dist
import networkx as nx
import pickle
import torch

## Simulate some data
class DataSimulator:
    def __init__(self):
        self.number_nodes = None
        self.network = None
        self.sim_parameters = None
        self.protein_mu = None
        self.ordered_nodes = None
        self.network_seed = None

    def generate_graph(self, n):

        seed = random.randint(0,10000)
        if n > 2:
            nodes=0
            while nodes < n:
                connected_graph = nx.gnp_random_graph(n, 0.25, directed=True, seed=seed)
                dag = nx.DiGraph([(u,v,{'weight' : np.random.randint(-10,10)}) for (u,v)
                                  in connected_graph.edges() if u<v])
                self.network_seed = seed
                seed = random.randint(0, 10000)
                nodes = len(dag.nodes)
        else:
            connected_graph = nx.gnp_random_graph(n, 1., directed=True, seed=seed)
            dag = nx.DiGraph([(u, v, {'weight': np.random.randint(-10, 10)}) for (u, v)
                              in connected_graph.edges() if u < v])
            self.network_seed = seed

        self.number_nodes = len(dag.nodes)
        self.network = dag

    def ground_truth_values(self):

        protein_mean = dict()

        beta_list = dict()

        nodes = list(nx.topological_sort(self.network))
        self.ordered_nodes = nodes
        for i in range(len(nodes)):

            intercept = dist.Uniform(15., 30.).sample()

            in_edges = list(self.network.in_edges(nodes[i]))

            if len(in_edges) == 0:
                mu = intercept
            else:
                temp_beta_list = list()
                coef_list = list()
                for j in range(len(in_edges)):
                    coef = torch.tensor(.3)#dist.Uniform(-1., 1.).sample()
                    coef_list.append(coef)
                    temp_beta_list.append(coef*protein_mean[in_edges[j][0]])

                mu = intercept + sum(np.array(temp_beta_list))
                beta_list[nodes[i]] = coef_list

            protein_mean[nodes[i]] = mu

        self.beta_list = beta_list
        self.protein_mu = protein_mean

    def id_no_impute(self, data, parameters):

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

    def simulate_data(self,
                      num_proteins,
                      num_bio_reps,
                      num_tech_reps,
                      num_features,
                      num_conditions,
                      include_missing=False,
                      mar_thresh=.05,
                      identify_nonimputable=False):

        """
        Function to simulate data with random effects given a specified experimental design.

        :param num_proteins: Number of proteins to simulate. May be fewer due to graph creation.
        :param num_bio_reps: Number of biological replicates to simulate.
        :param num_tech_reps: Number of technical replicates to simulate.
        :param num_features: Number of features per protein.
        :param num_conditions: Number of conditions.
        :param include_missing: Boolean on whether to mask some of the observed data. Masks with a combination of MNAR and MCAR.
        :param mar_thresh: MAR probability.
        :param identify_nonimputable run id_no_impute function or not. Only useful for missingess testing.
        :return:
        """

        ## Graph generation (can be ignored for this sim)
        self.generate_graph(num_proteins)

        ## Keep track of experimental design
        self.sim_parameters = {"Proteins": self.number_nodes,
                               "Bio_Replicate": num_bio_reps,
                               "Tech_Replicate": num_tech_reps,
                               "Features": num_features,
                               "Conditions" : num_conditions,
                               "MAR": mar_thresh}

        ## Create overall mean for each protein
        self.ground_truth_values()

        sim_data = pd.DataFrame()

        ## Generate data
        row = 0
        for n in range(self.number_nodes):

            ## TODO: All these effects are unique for each protein. We can make them the same for all proteins by putting outside for loop.
            ## Condition effect
            condition_effect = np.random.uniform(-3, 3, num_conditions)

            ## bio replicate effect
            bio_rep_effect = np.random.uniform(-2, 2, num_bio_reps)

            ## tech replicate effect
            tech_rep_effect = np.random.uniform(-2, 2, num_tech_reps)

            ## Overall sigma
            sigma = dist.Uniform(.1, .5).sample()

            for f in range(num_features):

                ## TODO: Putting feature here makes feature effect the same for all conditions/replicates
                ## Sample feature effect
                feat_effect = dist.Normal(0., .5).sample()

                for c in range(num_conditions):
                    for b in range(num_bio_reps):
                        for t in range(num_tech_reps):

                            ## Use linear combination to find overall mu
                            mu = self.protein_mu[self.ordered_nodes[n]] + condition_effect[c] + \
                                 bio_rep_effect[b] + tech_rep_effect[t] + feat_effect
                            ## Sample observed value
                            val = dist.Normal(mu, sigma).sample()

                            ## Add observation to data with all simulated effects to check ground truth
                            sim_data.loc[row, "Intensity"] = float(val)
                            sim_data.loc[row, "True_Intensity"] = float(val)
                            sim_data.loc[row, "Condition"] = c
                            sim_data.loc[row, "Bioreplicate"] = b
                            sim_data.loc[row, "Technical_Run"] = t
                            sim_data.loc[row, "Protein"] = self.ordered_nodes[n]
                            sim_data.loc[row, "Feature"] = f
                            sim_data.loc[row, "True_mean"] = float(mu)
                            sim_data.loc[row, "True_std"] = float(sigma)
                            sim_data.loc[row, "Protein_mean"] = self.protein_mu[self.ordered_nodes[n]].detach().numpy()
                            sim_data.loc[row, "Technical_Run_effect"] = tech_rep_effect[t]
                            sim_data.loc[row, "Bioreplicate_effect"] = bio_rep_effect[b]
                            sim_data.loc[row, "Condition_effect"] = condition_effect[c]
                            sim_data.loc[row, "Feature_effect"] = feat_effect.detach().numpy()

                            row += 1

        if include_missing:
            for i in range(len(sim_data)):
                mar_prob = np.random.uniform(0, 1)
                mnar_prob = np.random.uniform(0, 1)

                mnar_thresh = 1 / (1 + np.exp(-6.5 + (.4 * sim_data.loc[i, "True_Intensity"])))
                sim_data.loc[i, "MNAR_threshold"] = mnar_thresh

                if mar_prob < mar_thresh:
                    sim_data.loc[i, "Intensity"] = np.nan
                    sim_data.loc[i, "MAR"] = True
                else:
                    sim_data.loc[i, "MAR"] = False

                if mnar_prob < mnar_thresh:
                    sim_data.loc[i, "Intensity"] = np.nan
                    sim_data.loc[i, "MNAR"] = True
                else:
                    sim_data.loc[i, "MNAR"] = False

        sim_data.loc[:, "Missing"] = np.isnan(sim_data["Intensity"]) * 1
        if identify_nonimputable:
            sim_data = self.id_no_impute(sim_data, self.sim_parameters)

        self.data = sim_data

    def simulate_data_by_rep(self, num_proteins,num_conditions, num_reps, num_features, mar_thresh=.05):

        """
        Simulates data using a network and randomly sampled effects.

        :param num_proteins: Number of proteins to simulate.
        :param num_conditions:  Number of conditions to simulate.
        :param num_reps: Numer of replicates to simulate.
        :param num_features: Number of features per protein to simulate.
        :param mar_thresh: Missing at random probability.
        :return: Class with simulated data.
        """

        self.generate_graph(num_proteins)

        ## Get node order
        nodes = list(nx.topological_sort(self.network))
        self.ordered_nodes = nodes

        self.sim_parameters = {"Proteins": self.number_nodes,
                               "Conditions": num_conditions,
                               "Replicate": num_reps,
                               "Features": num_features,
                               "MAR": mar_thresh}

        sim_data = pd.DataFrame({"Protein": np.repeat(np.arange(0, self.number_nodes),
                                                        num_conditions * num_reps * num_features),
                                 "Condition": np.concatenate([np.repeat(np.arange(0, num_conditions),
                                                        num_reps * num_features) for _ in range(self.number_nodes)]),
                                 "Run": np.concatenate([np.repeat(np.arange(0, num_reps), num_features)
                                                          for _ in range(self.number_nodes * num_conditions)]),
                                 "Feature": np.concatenate([np.arange(0, num_features)
                                                        for _ in range(self.number_nodes * num_reps * num_conditions)])
                                 })

        ## Get protein mean
        protein_mean = np.random.uniform(15., 25., self.number_nodes)
        sim_data.loc[:, "Protein_mean"] = np.repeat(protein_mean, num_conditions * num_features * num_reps)

        # Add condition effect
        condition_effect = list()
        for i in range(self.number_nodes):
            overall_mean = sim_data.loc[sim_data["Protein"] == i]["Protein_mean"].unique()
            for j in range(num_conditions):
                if np.random.randint(0,1000) <= 25:
                    condition_effect.append(np.random.uniform(overall_mean*.15, overall_mean*.3))
                elif np.random.randint(0,1000) <= 50:
                    condition_effect.append(np.random.uniform(-overall_mean*.15, -overall_mean*.3))
                else:
                    condition_effect.append(np.random.uniform(-overall_mean*.025, overall_mean*.025))
        sim_data.loc[:, "Condition_effect"] = np.repeat(condition_effect, num_reps * num_features)

        ## replicate effect
        ## Generate run effect
        rep_effect = list()
        for i in range(self.number_nodes):
            overall_mean = sim_data.loc[sim_data["Protein"] == i]["Protein_mean"].unique()
            for j in range(num_reps*num_conditions):
                rep_effect.append(np.random.uniform(-overall_mean*.1, overall_mean*.1))

        sim_data.loc[:, "Run_effect"] = np.repeat(rep_effect, num_features)

        run_effect = pd.DataFrame()
        run_effect_row = 0
        coef_list = dict()

        tracker = 0
        for condition_idx in range(num_conditions):
            for replicate_idx in range(num_reps):

                for protein_idx in nodes:
                    in_edges = list(self.network.in_edges(protein_idx))
                    # protein_sigma=2.
                    # replicate_mean = np.random.normal(sim_data.loc[sim_data["Protein"] == protein_idx]
                    #                                   ["Protein_mean"].unique(), protein_sigma)[0]
                    replicate_mean = sim_data.loc[sim_data["Protein"] == protein_idx]["Protein_mean"].unique()[0]
                    # if len(in_edges) > 0:
                    #     replicate_mean=0

                    coef_temp = np.nan
                    if len(in_edges) > 0:
                        coef_temp = list()
                        for j in range(len(in_edges)):
                            coef = dist.Uniform(-.15, .15).sample()#torch.tensor(-.5)
                            coef_temp.append(coef)

                            protein_effect = coef * run_effect.loc[(run_effect["Protein"] == in_edges[j][0]) &
                                                                   (run_effect["Run"] == replicate_idx) &
                                                                   (run_effect["Condition"] == condition_idx)][
                                "Run_Protein_mean"].values[0]

                            replicate_mean += protein_effect.detach().numpy()
                    # else:

                    run_change = sim_data.loc[(sim_data["Protein"] == protein_idx) &
                                              (sim_data["Run"] == replicate_idx)]["Run_effect"].unique()[0]
                    condition_change = sim_data.loc[(sim_data["Protein"] == protein_idx) &
                                                    (sim_data["Condition"] == condition_idx)]["Condition_effect"].unique()[0]

                    replicate_mean += run_change
                    replicate_mean += condition_change

                    coef_list[protein_idx] = coef_temp
                    run_effect.loc[run_effect_row, "Protein"] = protein_idx
                    run_effect.loc[run_effect_row, "Run"] = replicate_idx
                    run_effect.loc[run_effect_row, "Condition"] = condition_idx
                    run_effect.loc[run_effect_row, "Run_Protein_mean"] = replicate_mean
                    run_effect_row += 1

                    tracker += 1
                    if tracker % 1000 == 0:
                        print(tracker)

        self.coef_list = coef_list

        ## Join in final protein values
        sim_data = pd.merge(sim_data, run_effect, how="left", on=["Protein", "Condition", "Run"])

        ## Create feature effects
        feat_effect = list()
        for i in range(len(nodes)):
            overall_mean = sim_data.loc[sim_data["Protein"] == i]["Protein_mean"].unique()
            temp_vals = np.random.normal(0., overall_mean*.05, num_features*num_conditions)
            temp_feat_effect = np.concatenate([temp_vals for _ in range(num_reps)])
            feat_effect.append(temp_feat_effect)
        sim_data.loc[:, "Feature_effect"] = np.concatenate(feat_effect)

        ## apply feature effect for final result
        sigma = np.random.uniform(.1, .3)
        sim_data.loc[:, "True_std"] = sigma

        for i in range(len(sim_data)):
            temp_mu = sim_data.loc[i, "Run_Protein_mean"] + sim_data.loc[i, "Feature_effect"]
            observed_value = np.random.normal(temp_mu, sigma)

            sim_data.loc[i, "Intensity"] = observed_value
            sim_data.loc[i, "True_Intensity"] = observed_value

        ## Mask some missing data
        for i in range(len(sim_data)):
            mar_prob = np.random.uniform(0, 1)
            mnar_prob = np.random.uniform(0, 1)

            mnar_thresh = 1 / (1 + np.exp(-3 + (.4 * sim_data.loc[i, "True_Intensity"])))
            sim_data.loc[i, "MNAR_threshold"] = mnar_thresh

            if mar_prob < mar_thresh:
                sim_data.loc[i, "Intensity"] = np.nan
                sim_data.loc[i, "MAR"] = True
            else:
                sim_data.loc[i, "MAR"] = False

            if mnar_prob < mnar_thresh:
                sim_data.loc[i, "Intensity"] = np.nan
                sim_data.loc[i, "MNAR"] = True
            else:
                sim_data.loc[i, "MNAR"] = False

        sim_data.loc[:, "Missing"] = np.isnan(sim_data["Intensity"]) * 1

        sim_data = self.id_no_impute(sim_data, self.sim_parameters)
        self.data = sim_data

def main():
    simulator = DataSimulator()
    n=250
    simulator.simulate_data_by_rep(n, 4, 3, 10, mar_thresh=0.03)
    # simulator.simulate_data(5, 4, 2, 10, 4)
    for i in range(n):
        print(simulator.data[simulator.data["Protein"] == i]["Missing"].sum())

    ## Save whole obj to pickle
    with open(r"data/simulated_data_25.pickle", "wb") as output_file:
        pickle.dump(simulator, output_file)

    ## Save data to csv
    # simulator.data.to_csv("simulated_data.csv", index=False)

if __name__ == "__main__":
    main()