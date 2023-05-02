
import pandas as pd
import numpy as np

import copy
import pickle

from simulation_code import DataSimulator
from utils import compare_to_gt

from sklearn.impute import KNNImputer

class DefaultMethods:

    def __init__(self, method, imputation=None):

        self.method = method
        self.summarized_data = None
        self.imputation = imputation
        self.imputed_data = None

    def impute(self, data):

        if self.imputation == "KNN":

            protein_data = data.loc[:, ["Intensity", "Run", "Protein", "Feature"]]

            imputer = KNNImputer(n_neighbors=20)
            imputed_data = imputer.fit_transform(protein_data)

            imputed_df = pd.DataFrame({"Intensity" : imputed_data[:,0],
                                       "Run" : imputed_data[:,1],
                                       "Protein" : imputed_data[:,2],
                                       "Feature" : imputed_data[:,3]}
            )

            self.imputed_data = imputed_df


    def tukey_median_polish(self, data, eps = 0.01, maxiter=10, trace_iter=True, na_rm=True):

        z = copy.copy(data)
        nr = data.shape[0]
        nc = data.shape[1]
        t = 0
        oldsum = 0

        r = np.array([0 for _ in range(nr)])
        c = np.array([0 for _ in range(nc)])

        for iter in range(maxiter):
            rdelta = list()
            if na_rm:
                for i in range(nr):
                    rdelta.append(np.nanmedian(z[i, :]))
            else:
                for i in range(nr):
                    rdelta.append(np.median(z[i, :]))
            rdelta = np.array(rdelta)

            z = z - np.repeat(rdelta, nc, axis=0).reshape(nr, nc)
            r = r + rdelta
            if na_rm:
                delta = np.nanmedian(c)
            else:
                delta = np.median(c)
            c = c - delta
            t = t + delta

            cdelta = list()
            if na_rm:
                for i in range(nc):
                    cdelta.append(np.nanmedian(z[:, i]))
            else:
                for i in range(nc):
                    cdelta.append(np.median(z[:, i]))
            cdelta = np.array(cdelta)

            z = z - np.repeat(cdelta, nr, axis=0).reshape(nr, nc, order='F')
            c = c + cdelta

            if na_rm:
                delta = np.nanmedian(r)
            else:
                delta = np.median(r)

            r = r - delta
            t = t + delta

            if na_rm:
                newsum = np.nansum(abs(z))
            else:
                newsum = np.sum(abs(z))

            converged = (newsum == 0) | (abs(newsum - oldsum) < eps * newsum)
            if converged:
                break
            oldsum = newsum
            # if trace_iter:
            #     print("{0}: {1}\n".format(str(iter), str(newsum)))

        ## TODO Add in converged info
        # if (converged) {
        # if (trace.iter)
        #   cat("Final: ", newsum, "\n", sep = "")
        # }
        # else warning(sprintf(ngettext(maxiter, "medpolish() did not converge in %d iteration",
        # "medpolish() did not converge in %d iterations"), maxiter),
        # domain = NA)

        ans = {"overall": t, "row": r, "col": c, "residuals": z}
        return ans
    def summarization(self, data):

        if self.method == "TMP":
            runs = data["Run"].unique()
            proteins = data["Protein"].unique()
            result_list = list()

            for i in proteins:

                protein_data = data[data["Protein"] == i]
                protein_data = protein_data.loc[:, ['Intensity', 'Run', 'Feature']]
                protein_data = protein_data.pivot(index="Feature", columns="Run", values="Intensity").values
                tmp_results = self.tukey_median_polish(protein_data)
                tmp_results = pd.DataFrame(data=tmp_results['overall'] + tmp_results['col'].reshape(1, len(runs)),
                                           columns=runs)
                tmp_results.loc[0, "Protein"] = i
                result_list.append(tmp_results)

            result_df = pd.concat(result_list)

        if self.method == "mean":
            runs = data["Run"].unique()
            proteins = data["Protein"].unique()
            result_list = list()

            for i in proteins:
                protein_data = data[data["Protein"] == i]
                protein_data = protein_data.loc[:, ['Intensity', 'Run', 'Feature']]
                protein_data = protein_data.pivot(index="Feature", columns="Run", values="Intensity")

                tmp_results = protein_data.mean(axis=0, skipna=True).values
                tmp_results = pd.DataFrame(data=tmp_results.reshape(1,len(runs)), columns=runs)
                tmp_results.loc[0, "Protein"] = i
                result_list.append(tmp_results)

            result_df = pd.concat(result_list)

        self.summarized_data = result_df

def main():
    with open(r"data/simulated_data.pickle", "rb") as input_file:
        simulator = pickle.load(input_file)

    method = DefaultMethods("TMP", "KNN")
    method.impute(simulator.data)
    method.summarization(method.imputed_data)
    comparison = compare_to_gt(method.summarized_data, simulator.data)
    print(comparison["full"])
    print(comparison["differences"])
    print(comparison["total"])

    method = DefaultMethods("TMP")
    method.summarization(simulator.data)
    comparison = compare_to_gt(method.summarized_data, simulator.data)
    print(comparison["full"])
    print(comparison["differences"])
    print(comparison["total"])
    #
    # method = DefaultMethods("mean")
    # method.summarization(simulator.data)
    # comparison = compare_to_gt(method.summarized_data, simulator.data)
    # print(comparison["full"])
    # print(comparison["differences"])
    # print(comparison["total"])

if __name__ == "__main__":
    main()
