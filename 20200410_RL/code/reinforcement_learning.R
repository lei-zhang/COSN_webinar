# =============================================================================
#### Info #### 
# =============================================================================
# simple reinforcement learning model
#
# Lei Zhang
# lei.zhang@univie.ac.at


# =============================================================================
#### Construct Data #### 
# =============================================================================
# clear workspace
library(rstan)
library(ggplot2)

load('rl_sp_ss.RData')
sz = dim(rl_ss)
nTrials = sz[1]

dataList = list(nTrials=nTrials, 
                 choice=rl_ss[,1], 
                 reward=rl_ss[,2])

# =============================================================================
#### Running Stan #### 
# =============================================================================
rstan_options(auto_write = TRUE)
options(mc.cores = 4)

modelFile = 'rw.stan'

nIter     = 2000
nChains   = 4 
nWarmup   = floor(nIter/2)
nThin     = 1

cat("Estimating", modelFile, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")

fit_rl = rstan::stan(modelFile, 
               data    = dataList, 
               chains  = nChains,
               iter    = nIter,
               warmup  = nWarmup,
               thin    = nThin,
               init    = "random")

cat("Finishing", modelFile, "model simulation ... \n")
endTime = Sys.time(); print(endTime)  
cat("It took",as.character.Date(endTime - startTime), "\n")

# =============================================================================
#### Model Summary and Diagnostics #### 
# =============================================================================
print(fit_rl, pars = c('alpha','tau'))

stan_trace(fit_rl, pars = c('alpha','tau'), inc_warmup = F)
stan_plot(fit_rl, pars=c('alpha','tau'), show_density=T, fill_color = 'skyblue')

# =============================================================================
#### Fitting multiple subjects with hBayesDM #### 
# =============================================================================
library(hBayesDM)
fit_rw = hBayesDM::bandit2arm_delta(data = 'example', 
                                    niter = 2000, 
                                    nwarmup = 1000,
                                    adapt_delta = .8)

















