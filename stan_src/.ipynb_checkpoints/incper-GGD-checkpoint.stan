functions {
  real gengammaloc(real a, real sigma, real logmean) {
    real q = inv_square(a), log_q = log(q), cinv = sigma / a,
      value = lgamma(q + cinv) - lgamma(q) - log_q * cinv;
        
    return logmean - value;
  }
  
  real gengammacv(real a, real mu, real sigma) {
    real q = inv_square(a), cinv = sigma / a,
        value = lgamma(q) + lgamma(2 * cinv + q) - 2 * lgamma(cinv + q); 

    return sqrt(expm1(value));
  }

  real gengamma_lpdf(vector x, real a, real mu, real sigma) {
    int N = size(x);
    vector[N] logx = log(x),
        z = (logx - mu) / sigma;
    real q = inv_square(a);
    vector[N] y = q * exp(a .* z);

    return gamma_lpdf(y | q, 1) + sum(a * z - logx) - N * (log(sigma) + log(a));
  }

  real gengamma_cdf(real x, real a, real mu, real sigma) {
    real logx = log(x),
        z = (logx - mu) / sigma,
        q = inv_square(a);

    return gamma_cdf(q * exp(a * z) | q, 1);
  }

  real gengamma_lcdf(real x, real a, real mu, real sigma) {
    real logx = log(x),
        z = (logx - mu) / sigma,
        q = inv_square(a),
        value = gamma_lcdf(q * exp(a * z) | q, 1);

    return value;
  }

  real gengamma_lcdf(vector x, real a, real mu, real sigma) {
    int N = size(x);
    vector[N] logx = log(x),
        z = (logx - mu) / sigma;
    real q = inv_square(a),
        value = gamma_lcdf(q * exp(a .* z) | q, 1);

    return value;
  }
}

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

  real jitter = 1.0e-8;
}

parameters {
    vector<lower = 0, upper = 1>[N] exposure_raw, onset_raw;
    vector<lower = 0>[N_cens] negative_exposureL_cens;
  
    real loga, logsigma, logmean_interval;
}

transformed parameters {
    real<lower = 0> a = exp(loga),
        q = inv_square(a), 
        sigma = exp(logsigma), 
        mean_interval = exp(logmean_interval);
    real logq = - 2.0 * loga; 
    real mu = gengammaloc(a, sigma, logmean_interval);
    real<lower = 0> sd_interval = mean_interval * gengammacv(a, mu, sigma);
    real logsd_interval = log(sd_interval);

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
}

model {
    logq ~ normal(mean_prior, sd_prior);
    logsd_interval ~ normal(mean_prior, sd_prior);
    logmean_interval ~ normal(mean_prior, sd_prior);
    // Jacobian adjustment
    target += log(2 * sigma / a) + log1p(square(mean_interval / sd_interval)) 
               + log(abs(digamma(q + 2 * sigma / a) - digamma(q + sigma / a)));

    onset_raw ~ beta(1, 1); 
    exposure_raw ~ beta(1, 1);

    negative_exposureL_cens ~ exponential(0.1);

    vector[N] exposure = fma(to_vector(exposureR) - to_vector(exposureL_realized), exposure_raw, to_vector(exposureL_realized)),
            onset;
    for (n in 1 : N) {
        real onsetL_ = exposure[n] > onsetL[n] ? exposure[n] : onsetL[n];
        onset[n] = fma(onsetR[n] - onsetL_, onset_raw[n], onsetL_);
    }
    vector[N] time_interval = onset - exposure,
        time_interval_from_truncation = to_vector(truncation_day) + 1 - exposure; 
    
    target += gengamma_lpdf(time_interval | a, mu, sigma) 
            - gengamma_lcdf(time_interval_from_truncation | a, mu, sigma); // truncation
}

generated quantities {
    real pred_interval;
    {
      real y = gamma_rng(q, 1),
        logx = mu + sigma / a * log(y / q);
      pred_interval = exp(logx);
    }

    vector[M + 1] cdf = rep_vector(0.0, M + 1), pdf = rep_vector(0.0, M + 1), x;
    {
        for (i in 1 : M + 1) {
            x[i] = incper_max * (i - 1 + jitter) / (M + jitter);
            cdf[i] = gengamma_cdf(x[i] | a, mu, sigma);
        }
        pdf[2 : M + 1] = tail(cdf, M) - head(cdf, M);
        pdf[1] = cdf[1];
    }
}