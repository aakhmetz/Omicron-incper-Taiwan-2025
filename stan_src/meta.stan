data {
  int<lower=1> J;  // number of studies with available data
  vector[J] mu_known;  // means of known studies
  vector<lower = 0>[J] stderr_known;  // standard errors of known studies
}

parameters {
  real mu_raw;
  real<lower = 0> tau;
}

transformed parameters {
  real mu = 8 * mu_raw + 4;
}

model {
  // Priors
  mu_raw ~ std_normal();  // weakly informative prior for the overall mean
  tau ~ cauchy(0, 0.5);  // weakly informative prior for the between-study variability

  // Likelihood
  mu_known ~ normal(mu, sqrt(square(stderr_known) + square(tau)));
}

generated quantities {
  real mu_pred, tau_squared;

    {
        // Calculate tau-squared statistic
        tau_squared = square(tau);

        // Predict a future observation for a hypothetical new study
        mu_pred = normal_rng(mu, sqrt(tau));
    }
}