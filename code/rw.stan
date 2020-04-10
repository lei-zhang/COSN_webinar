data{
    int nTrials;
    int<lower=1,upper=2> choice[nTrials];
    int<lower=-1,upper=1> reward[nTrials];
}

parameters {
    real<lower=0,upper=1> alpha; // learning rate
    real<lower=0,upper=20> tau;  // softmax inverse tem
}

model {
    vector[2] v;
    real pe;
    
    v = rep_vector(0, 2);

    for (t in 1:nTrials) {
        // choice[t] ~ categorical(softmax(tau*v));
        choice[t] ~ categorical_logit(tau*v);
        
        pe = reward[t] - v[choice[t]]; // prediction error
        v[choice[t]] = v[choice[t]] + alpha * pe; // value update
    }
}

generated quantities {
    vector[2] v[nTrials];
    real pe[nTrials];
    int  y_pred[nTrials];
    real v_chn[nTrials];
    real acc[nTrials];
    real log_lik;
    
    v[1] = rep_vector(0, 2);
    log_lik = 0;
    
    for (t in 1:nTrials) {
        // y_pred[t] = categorical_rng(softmax(tau*v[t]));
        y_pred[t] = categorical_logit_rng(tau*v[t]);
        log_lik = log_lik + categorical_logit_lpmf(choice[t] | tau*v[t]); // --> p(D|theta)
        
        v_chn[t] = v[t, choice[t]];           
        acc[t] = (y_pred[t] == choice[t]) * 1.0;             
        pe[t] = reward[t] - v[t, choice[t]]; // prediction error
        
        if (t < nTrials) {
            v[t+1,choice[t]] = v[t,choice[t]] + alpha * pe[t]; // value update
            v[t+1,3-choice[t]] = v[t,3-choice[t]]; 
        }
    }
}
