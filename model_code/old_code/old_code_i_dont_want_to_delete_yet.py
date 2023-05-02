########################################################################################################################
# Independent models assuming same dimension of all proteins
########################################################################################################################
def regression_model(data=None, num_proteins=10, num_features = 9, num_runs = 2):

    ## MAR indicator
    alpha_q = pyro.sample("alpha_q", dist.Normal(10., .01))
    beta_q = pyro.sample("beta_q", dist.Normal(100., .1))

    # mar_prob = pyro.sample("MAR", dist.Beta(alpha, beta))
    mar = pyro.sample("MAR", dist.Beta(alpha_q, beta_q))

    beta0_mu = pyro.sample("beta0_mu", dist.Normal(torch.tensor(10.), torch.tensor(.01)))
    beta1_mu = pyro.sample("beta1_mu", dist.Normal(torch.tensor(.5), torch.tensor(.001)))

    beta0 = pyro.sample("beta0", dist.Normal(beta0_mu, torch.tensor(.01)))
    beta1 = pyro.sample("beta1", dist.Normal(beta1_mu, torch.tensor(.0001)))

    for protein in pyro.plate("protein", num_proteins):

        mu = pyro.sample("mu_{0}".format(protein), dist.Uniform(5. ,40.))
        sigma = pyro.sample("sigma_{0}".format(protein),
                            dist.Uniform(0. ,100.))

        with pyro.plate("runs_{}".format(protein), num_runs):
            br = pyro.sample("bR_{0}".format(protein),
                             dist.Normal(torch.zeros(num_runs), torch.ones(num_runs)))
        with pyro.plate("features_{}".format(protein), num_features):
            bf = pyro.sample("bF_{0}".format(protein),
                             dist.Normal(torch.zeros(num_features), torch.ones(num_features)))

        mean = mu + torch.sum(br  * data[protein, : ,3:3+num_runs], dim=1) + \
               torch.sum(bf * data[protein, : ,3+num_runs:], dim=1)

        with pyro.plate("data_{0}".format(protein), len(data[protein, :, 0])):
            mnar = 1/ (1 + torch.exp(-beta0 + (beta1 * mean)))
            missing_prob = mar * ((1 - mar) * mnar)

            missing = pyro.sample("missing_{0}".format(protein),
                                  dist.Bernoulli(probs=missing_prob),
                                  obs=data[protein, :, 1])
            with poutine.mask(mask=((missing == 0))):
                pyro.sample("obs_{0}".format(protein),
                            dist.Normal(mean, sigma), obs=data[protein, :, 0])

def regression_guide(data=None, num_proteins=10, num_features=9, num_runs=2):

    ## MAR
    alpha_q_mu = pyro.param("alpha_q_mu", torch.tensor(10.0),
                            constraint=constraints.positive)
    alpha_q_std = pyro.param("alpha_q_std", torch.tensor(1.),
                             constraint=constraints.positive)
    alpha_q = pyro.sample("alpha_q", dist.Normal(alpha_q_mu, alpha_q_std))

    beta_q_mu = pyro.param("beta_q_mu", torch.tensor(100.0),
                           constraint=constraints.positive)
    beta_q_std = pyro.param("beta_q_std", torch.tensor(5.),
                            constraint=constraints.positive)
    beta_q = pyro.sample("beta_q", dist.Normal(beta_q_mu, beta_q_std))

    pyro.sample("MAR", dist.Beta(alpha_q, beta_q))

    beta0_mu_mu = pyro.param("beta0_mu_mu", torch.tensor(10.0))
    beta0_mu_std = pyro.param("beta0_mu_std", torch.tensor(1.),
                              constraint=constraints.positive)
    beta0_mu = pyro.sample("beta0_mu", dist.Normal(beta0_mu_mu, beta0_mu_std))

    beta1_mu_mu = pyro.param("beta1_mu_mu", torch.tensor(.5))
    beta1_mu_std = pyro.param("beta1_mu_std", torch.tensor(.1),
                              constraint=constraints.positive)
    beta1_mu = pyro.sample("beta1_mu", dist.Normal(beta1_mu_mu, beta1_mu_std))

    pyro.sample("beta0", dist.Normal(beta0_mu, torch.tensor(.01)))
    pyro.sample("beta1", dist.Normal(beta1_mu, torch.tensor(.0001)))

    for protein in pyro.plate("protein", num_proteins):

        ## Sample mean
        mu_loc = pyro.param("mu_loc_{}".format(protein), torch.tensor(20.))
        mu_scale = pyro.param("mu_scale_{}".format(protein), torch.tensor(2.),
                              constraint=constraints.positive)
        pyro.sample("mu_{}".format(protein), dist.Normal(mu_loc, mu_scale))

        ## Sample std
        sigma_loc = pyro.param("sigma_loc_{}".format(protein), torch.tensor(0.))
        sigma_scale = pyro.param("sigma_scale_{}".format(protein), torch.tensor(.25),
                                 constraint=constraints.positive)
        pyro.sample("sigma_{}".format(protein),
                            dist.LogNormal(sigma_loc, sigma_scale))

        ## Coefficients
        br_loc = pyro.param("br_loc_{0}".format(protein), torch.randn(num_runs))
        br_scale = pyro.param("br_scale_{0}".format(protein), torch.ones(num_runs),
                              constraint=constraints.positive)

        bf_loc = pyro.param("bf_loc_{0}".format(protein), torch.randn(num_features))
        bf_scale = pyro.param("bf_scale_{0}".format(protein), torch.ones(num_features),
                              constraint=constraints.positive)

        with pyro.plate("runs_{}".format(protein), num_runs):
            pyro.sample("bR_{0}".format(protein),
                             dist.Normal(br_loc, br_scale))
        with pyro.plate("features_{}".format(protein), num_features):
            pyro.sample("bF_{0}".format(protein),
                             dist.Normal(bf_loc, bf_scale))