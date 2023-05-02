
import pandas as pd

def compare_to_gt(summarized, sim_data):
    sim_data_format = sim_data.loc[:, ["Protein", "Run", "Run_Protein_mean"]].drop_duplicates().pivot(
        index="Protein", columns="Run", values="Run_Protein_mean"
    ).reset_index()

    comparison_dict = dict()
    comparison = pd.merge(summarized, sim_data_format, how="outer", on="Protein")

    for i in sim_data["Run"].unique():
        comparison.loc[:, "{}".format(i)] = comparison.loc[:, "{}_x".format(i)
                                            ] - comparison.loc[:, "{}_y".format(i)]

    comparison_dict["full"] = comparison
    comparison_dict["differences"] = comparison.loc[:, ["Protein", "0", "1", "2", "3", "4", "5"]]
    comparison = comparison.set_index("Protein")
    comparison_dict["total"] = abs(comparison.loc[:, ["0", "1", "2", "3", "4", "5"]]).sum(axis=1)
    comparison_dict["mean"] = abs(comparison.loc[:, ["0", "1", "2", "3", "4", "5"]]).mean(axis=1)

    return comparison_dict