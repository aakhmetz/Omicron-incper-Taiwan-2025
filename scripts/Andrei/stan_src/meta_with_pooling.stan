data {
  int<lower=1> J;  // number of studies with available data
  int<lower=1> V; // number of variants
  array[J] real mu_known;  // means of known studies
  array[J] real<lower = 0> stderr_known;  // standard errors of known studies
  array[J] int<lower = 1, upper = V> variant;
}

parameters {
  vector[V] mu_raw;
  vector<lower = 0>[V] tau;

  real mu_overall_raw;
  real<lower = 0> tau_overall;
}

transformed parameters {
  real mu_overall = 8 * mu_overall_raw + 4;
  vector[V] mu = tau_overall * mu_raw + mu_overall;
}

model {
  // Priors
  mu_raw ~ std_normal();
  tau ~ cauchy(0, 0.5);  // weakly informative prior for the between-study variability

  mu_overall_raw ~ std_normal();
  tau_overall ~ cauchy(0, 0.5);

  // Likelihood
  mu_known ~ normal(mu[variant], sqrt(square(to_vector(stderr_known)) + square(tau[variant])));
}

generated quantities {
  real tau_squared_overall;
  vector[V] tau_squared;

    {
        // Calculate tau-squared statistic
        tau_squared = square(tau);
        tau_squared_overall = square(tau_overall);

    }
}