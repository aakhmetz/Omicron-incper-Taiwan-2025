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
}

parameters {
    vector<lower = 0, upper = 1>[N] exposure_raw, onset_raw;
    vector<lower = 0>[N_cens] negative_exposureL_cens;
  
    real logmean_interval_raw, logparam1;
}

transformed parameters {
    real logmean_interval = logmean_interval_raw * sd_prior + mean_prior;
    real<lower = 0> mean_interval = exp(logmean_interval),
        param1 = exp(logparam1), 
        param2 = mean_interval / tgamma(1.0 + 1.0 / param1),
        sd_interval = param2 * sqrt(tgamma(1.0 + 2.0 / param1) - square(tgamma(1.0 + 1.0 / param1)));
    real logsd_interval = log(sd_interval);
}

model {
    logsd_interval ~ normal(mean_prior, sd_prior);
    logmean_interval_raw ~ normal(mean_prior, sd_prior);
    target += 2 * log(param2) - 2 * logsd_interval - 2 * log(param1) + 
        log(abs(square(tgamma(1.0 + 1.0 / param1)) * digamma(1.0 + 1.0 / param1) - tgamma(1.0 + 2.0 / param1) * digamma(1.0 + 2.0 / param1)));
  
    onset_raw ~ beta(1, 1); 
    exposure_raw ~ beta(1, 1);

    negative_exposureL_cens ~ exponential(0.1);

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
    
    target += weibull_lupdf(time_interval | param1, param2) 
                - weibull_lcdf(time_interval_from_truncation | param1, param2); // truncation
}

generated quantities {
    real pred_interval = weibull_rng(param1, param2);

    vector[M] cdf = rep_vector(0.0, M), pdf = rep_vector(0.0, M), x_cdf;
    {
        for (i in 1 : M) {
            x_cdf[i] = incper_max * i / M;
            cdf[i] = weibull_cdf(x_cdf[i] | param1, param2);
        }
        pdf[2 : M] = tail(cdf, M - 1) - head(cdf, M - 1);
        pdf[1] = cdf[1];
    }
}