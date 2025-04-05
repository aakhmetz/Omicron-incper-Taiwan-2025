data {
    int<lower=1> N; // number of case records 
    array[N] real onsetR;
    array[N] real<upper = onsetR> onsetL;
    array[N] real<upper = onsetR> exposureR;
    array[N] real<upper = exposureR> exposureL;
    array[N] int<lower = 0, upper = 1> censored;
    array[N] real<lower = onsetR> truncation_day;
    
    real mean_prior;
    real<lower = 0> sd_prior;

    // for generated quantities
    int<lower = 1> M;
    real incper_max;
}

transformed data {
  // calculate number of censored records
  int N_cens = sum(censored);

  int D = 3; // number of distributions for incper (Gamma, Weibull, Lognormal)

  real jitter = 1.0e-8;
}

parameters {
    vector<lower = 0, upper = 1>[N] exposure_raw, onset_raw;
    vector<lower = 0>[N_cens] negative_exposureL_cens;
  
    real logmean_interval_raw, logparam1_Weibull;

    simplex[D] weight; // mixing proportions
}

transformed parameters {
    real logmean_interval = logmean_interval_raw * sd_prior + mean_prior,
        logsd_interval;
    real<lower = 0> mean_interval = exp(logmean_interval), sd_interval;

    vector[D] param1, param2;
    {
        // Weibull distribution
        param1[2] = exp(logparam1_Weibull); 
        param2[2] = mean_interval / tgamma(1.0 + 1.0 / param1[2]);
        sd_interval = param2[2] * sqrt(tgamma(1.0 + 2.0 / param1[2]) - square(tgamma(1.0 + 1.0 / param1[2])));
        logsd_interval = log(sd_interval);

        // Gamma distribution
        param1[1] = square(mean_interval / sd_interval);
        param2[1] = mean_interval / square(sd_interval);

        // lognormal distribution
        param2[3] = sqrt(log(square(sd_interval / mean_interval) + 1.0));
        param1[3] = log(mean_interval) - square(param2[3]) / 2.0;        
    }

    vector[D] lps = log(weight); // internal component likelihoods
    {
        array[N] real exposureL_realized;
        {
          int index_censored = 0;
          for (n in 1 : N)
            if (censored[n]) {
              index_censored += 1;
              exposureL_realized[n] = exposureR[n] - negative_exposureL_cens[index_censored];
            } else {
              exposureL_realized[n] = exposureL[n];
            }
        }

        vector[N] exposure = fma(to_vector(exposureR) - to_vector(exposureL_realized), exposure_raw, to_vector(exposureL_realized)),
            onset;
        for (n in 1:N) {
            real onsetL_ = exposure[n] > onsetL[n] ? exposure[n] : onsetL[n];
            onset[n] = fma(onsetR[n] - onsetL_, onset_raw[n], onsetL_);
        }
        vector[N] time_interval = onset - exposure,
            time_interval_from_truncation = to_vector(truncation_day) + 1 - exposure; 

        lps[1] += gamma_lpdf(time_interval | param1[1], param2[1]) - gamma_lcdf(time_interval_from_truncation | param1[1], param2[1]);
        lps[2] += weibull_lpdf(time_interval | param1[2], param2[2]) - weibull_lcdf(time_interval_from_truncation | param1[2], param2[2]);
        lps[3] += lognormal_lpdf(time_interval | param1[3], param2[3]) - lognormal_lcdf(time_interval_from_truncation | param1[3], param2[3]);
    }
}

model {
    logsd_interval ~ normal(mean_prior, sd_prior);
    logmean_interval_raw ~ normal(mean_prior, sd_prior);
    target += 2 * log(param2[2]) - 2 * logsd_interval - 2 * log(param1[2]) + 
        log(abs(square(tgamma(1.0 + 1.0 / param1[2])) * digamma(1.0 + 1.0 / param1[2]) - tgamma(1.0 + 2.0 / param1[2]) * digamma(1.0 + 2.0 / param1[2])));
  
    onset_raw ~ beta(1, 1); 
    exposure_raw ~ beta(1, 1);

    negative_exposureL_cens ~ exponential(0.1);

    target += log_sum_exp(lps); // truncation
}

generated quantities {
    vector<lower = 0, upper = 1>[D] q = softmax(lps);
    int comp = categorical_rng(q);

    real pred_interval = (comp == 1) ? gamma_rng(param1[comp], param2[comp]) :
        ((comp == 2) ? weibull_rng(param1[comp], param2[comp]) : lognormal_rng(param1[comp], param2[comp]));

    vector[M + 1] cdf = rep_vector(0.0, M + 1), pdf = rep_vector(0.0, M + 1), x;
    {
        for (i in 1 : M + 1) {
            x[i] = incper_max * (i - 1 + jitter) / (M + jitter);
            cdf[i] = (comp == 1) ? weibull_cdf(x[i] | param1[comp], param2[comp]) :
                ((comp == 2) ? weibull_cdf(x[i] | param1[comp], param2[comp]) : lognormal_cdf(x[i] | param1[comp], param2[comp]));
     }
        pdf[2 : M + 1] = tail(cdf, M) - head(cdf, M);
        pdf[1] = cdf[1];
    }
}