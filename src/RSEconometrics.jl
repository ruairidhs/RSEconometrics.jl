module RSEconometrics

using LinearAlgebra,
      SparseArrays,
      Distributions,
      StaticArrays,
      ForwardDiff

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
       ml_varcov,                   # Maximum likelihood
       get_std_errs

end
