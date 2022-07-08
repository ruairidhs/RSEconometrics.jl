module RSEconometrics

using LinearAlgebra,
      SparseArrays,
      Distributions,
      StaticArrays,
      ForwardDiff,
      DiffRules,
      Optimization,
      OptimizationOptimJL

import Polynomials

include("kalman.jl")
include("arma.jl")
include("maximum_likelihood.jl")

export StateSpaceModel,             # Kalman filter
       KalmanPredictionIterator,
       kalman_log_likelihood,
       UnivariateARMA,              # ARMA
       construct_ma,
       construct_ar,
       arma_nll,
       get_constant,
       is_stationary,
       is_invertible,
       common_factors,
       estimate_arma,
       varcov_arma,
       ml_varcov,                   # Maximum likelihood
       get_std_errs

end
