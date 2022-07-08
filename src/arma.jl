# ====== Description =====
# Defines an ARMA type and related functions:
#   - Checks for stationarity, invertibility, common factors
#   - Simulation
#   - Maximum likelihood estimation via the Kalman Filter
"""
UnivariateARMA(Φ::AbstractVector, Θ::AbstractVector, μ::Number, d::Distribution)

Representation of the univariate ARMA(p, q) model:
``(1 - Φ_1 L - … - Φ_p L^p)(y_t - μ) = (1 + Θ_1 L + … + Θ_q L^q)ϵ_t``
where ``ϵ_t∼iid(d)``.

It is assumed that ``E(ϵ_t)=0``.
"""
struct UnivariateARMA{VΦ <: AbstractVector, VΘ <: AbstractVector,
                      T <: Number, D <: UnivariateDistribution}
    Φ::VΦ # different types to allow for static arrays of different lengths
    Θ::VΘ
    μ::T
    d::D
end

"""
    construct_ma(Θ; μ=0.0, σ2=1.0)
    construct_ar(Φ; μ=0.0, σ2=1.0)

Convenience functions to create UnivariateARMA objects with normal errors.
"""
function construct_ma(Θ; μ=0.0, σ2=1.0)
    return UnivariateARMA(eltype(Θ)[], Θ, μ, Normal(0, sqrt(σ2)))
end

"""
    construct_ma(Θ; μ=0.0, σ2=1.0)
    construct_ar(Φ; μ=0.0, σ2=1.0)

Convenience functions to create UnivariateARMA objects with normal errors.
"""
function construct_ar(Φ; μ=0.0, σ2=1.0)
    return UnivariateARMA(Φ, eltype(Φ)[], μ, Normal(0, sqrt(σ2)))
end

"""
    get_constant(arma::UnivariateARMA)

Compute `c` satisfying:
``(1-Φ_1 L - … - Φ_pL^p)y_t = c + (1 + Θ_1 L + … + Θ_q L^q)ϵ_t``
"""
get_constant(arma::UnivariateARMA) = (1 - sum(arma.Φ)) * arma.μ

"""
    is_stationary(arma::UnivariateARMA)

True if all of the roots of the AR polynomial are outside the unit circle.
"""
function is_stationary(arma)
    pn = Polynomials.Polynomial([1; -arma.Φ])
    return all(abs.(Polynomials.roots(pn)) .> 1)
end
DiffRules.@define_diffrule RSEconometrics.is_stationary(arma) = :(0)

"""
    is_invertible(arma::UnivariateARMA)

True if all of the roots of the MA polynomial are outside the unit circle.
"""
function is_invertible(arma::UnivariateARMA)
    pn = Polynomials.Polynomial([1; arma.Θ])
    return all(abs.(Polynomials.roots(pn)) .> 1)
end

"""
    common_factors(arma::UnivariateARMA)

True if the intersection of the AR and MA roots is not empty.
"""
function common_factors(arma::UnivariateARMA; tol=1e-6)
    ar_roots = Polynomials.roots(Polynomials.Polynomial([1; -arma.Φ]))
    ma_roots = Polynomials.roots(Polynomials.Polynomial([1; arma.Θ]))
    return mapreduce(t -> abs(t[1] - t[2]) < tol, |, Iterators.product(ar_roots, ma_roots))
end

"""
    StateSpaceModel(arma::UnivariateARMA)

Construct the state-space representation of `arma`.
"""
function StateSpaceModel(arma::UnivariateARMA{VΦ, VΘ, T, D}) where {VΦ <: AbstractArray,
                                                                    VΘ <: AbstractArray,
                                                                    T, D}
    # Some set up
    (; Φ, Θ, μ, d) = arma
    σ2 = var(d)
    p, q = map(length, (Φ, Θ))
    r = max(p, q + 1)

    # State equation
    F = diagm(-1 => ones(eltype(Φ), r-1))
    index = 1
    for value in Φ # don't use enumerate in case Φ has weird indices
        F[1, index] = value
        index += 1
    end
    Q = zeros(typeof(σ2), r, r)
    Q[1,1] = σ2

    At = μ * ones(1,1)
    Ht = zeros(eltype(Θ), 1, r)
    Ht[1] = 1
    index = 2
    for value in Θ
        Ht[index] = value
        index += 1
    end
    R = zeros(1,1)

    return StateSpaceModel(F, At, Ht, Q, R)
end

function arma_nll(u, params)
    p, q, ys = params
    arma = UnivariateARMA(u[1:p], u[p+1:p+q], u[p+q+1], Normal(0, max(u[p+q+2], 0)))
    return -kalman_log_likelihood(KalmanPredictionIterator(ys, StateSpaceModel(arma)))
end

"""
    rand(arma::UnivariateARMA, T::Int; burn_in::Int=0)

Draw a sample of length `T` from `arma`.

If `burn_in` is provided, `burn_in` number of draws are discarded before collecting the sample.
The sample is initiated with `y` values equal to `arma.μ`.
"""
function Base.rand(arma::UnivariateARMA, T::Int; burn_in::Int=0)
    p, q = map(length, (arma.Φ, arma.Θ))
    extΘ = [1; arma.Θ]

    ϵs = SVector{q+1}(rand(arma.d, q+1)) # [ϵ_t, ϵ_{t-1}, …, ϵ_{t-q}]; minimum length = 1
    ycache = zeros(SVector{p}) # [y_{t-1}, …, y_{t-p}]; minimum length = 0

    # need tailored indexing functions to generate static slices of static arrays
    ϵstart = SVector{q}(1:q)
    function shift_ϵs(ϵ, ϵs)
        return [ϵ; ϵs[ϵstart]]
    end

    ystart = p > 0 ? SVector{p-1}(1:p-1) : nothing
    function shift_ycache(y, yc)
        if length(yc) == 0
            return yc
        else
            return [y; yc[ystart]]
        end
    end

    function generate_y(ycache, ϵs)
        y = dot(ycache, arma.Φ) + dot(ϵs, extΘ)
        return y, shift_ycache(y, ycache), shift_ϵs(rand(arma.d), ϵs)
    end

    current_y = 0
    # do the burn-in
    for _ in 1:burn_in
        current_y, ycache, ϵs = generate_y(ycache, ϵs)
    end

    ys = zeros(T)
    for i in eachindex(ys)
        current_y, ycache, ϵs = generate_y(ycache, ϵs)
        ys[i] = current_y
    end

    return ys .+ arma.μ
end

# ===== Methods for the efficient construction of StaticArrays ARMA models =====
@generated function _make_F(arma)
    p = arma.parameters[1].parameters[4]
    q = arma.parameters[2].parameters[4]
    r = max(p, q+1)

    quote
        extended_Φ = hcat(permutedims(arma.Φ),
                          zeros(SMatrix{1, $(r-p)})
                         )
        diagm(Val(-1) => ones(SVector{$(r-1)})) + 
            vcat(extended_Φ, zeros(SMatrix{$(r-1), $r}))
    end
end

@generated function _make_Ht(arma)
    p = arma.parameters[1].parameters[4]
    q = arma.parameters[2].parameters[4]
    r = max(p, q+1)

    quote
        permutedims(vcat(SVector{1}(1), arma.Θ, zeros(SVector{$(r-q-1)})))
    end
end

@generated function _make_Q(arma)
    p = arma.parameters[1].parameters[4]
    q = arma.parameters[2].parameters[4]
    r = max(p, q+1)

    quote
        reshape(vcat(var(arma.d), zeros(SVector{$(r*r-1)})), Size($r, $r))
    end
end

function StateSpaceModel(arma::UnivariateARMA{VΦ, VΘ, T, D}) where {VΦ <: StaticArray,
                                                                    VΘ <: StaticArray,
                                                                    T, D}
    return StateSpaceModel(_make_F(arma),
                           arma.μ * SMatrix{1,1}(1),
                           _make_Ht(arma),
                           _make_Q(arma),
                           SMatrix{1,1}(zero(eltype(VΦ)))
                          )
end

# ===== Maximum likelihood estimation =====
function _static_arma_likelihood(Φ::SVector, Θ::SVector, μ, σ, ys;
        check_stationary=true, observation_likelihood=false
    )
    arma = UnivariateARMA(Φ, Θ, μ, Normal(0, sqrt(σ^2)))
    if check_stationary && !is_stationary(arma)
        return Inf
    end
    ssm = StateSpaceModel(arma)
    iter = KalmanPredictionIterator(ys, ssm)
    if observation_likelihood
        return map(t -> t.ll, iter)
    else
        return kalman_log_likelihood(iter)
    end
end

@generated function _split_arma_args(u, p, q)
    P, Q = p.parameters[1], q.parameters[1]
    Φ_indices = SVector{P}(1:P)
    Θ_indices = SVector{Q}((P+1):(P+Q))
    μ_index, σ_index = P + Q + 1, P + Q + 2
    quote
        (u[$Φ_indices], u[$Θ_indices], u[$μ_index], u[$σ_index])
    end
end

function _make_syms(p, q)
    ars = Symbol.("AR_" .* string(1:p))
    mas = Symbol.("MA_" .* string(1:q))
    return [ars; mas; :mu; :sigma]
end

function estimate_arma(ys, p, q; 
        init=[zeros(p); zeros(q); 0; 1],
        check_stationary=true,
        method=BFGS()
    )
    function obj(u, _)
        -_static_arma_likelihood(_split_arma_args(u, Val(p), Val(q))...,
                                 ys; check_stationary=check_stationary
                                )
    end

    prob = OptimizationProblem(OptimizationFunction(obj,
                                                    Optimization.AutoForwardDiff();
                                                    syms = _make_syms(p, q)
                                                   ),
                               init
                              )
    return solve(prob, method)
end

function varcov_arma(sol, ys, p, q; method=:sandwich)
    observation_likelihood = method != :hessian
    nll = (u, _) -> -_static_arma_likelihood(_split_arma_args(u, Val(p), Val(q))..., ys;
                                             check_stationary=false,
                                             observation_likelihood=observation_likelihood
                                            )
    return ml_varcov(nll, sol, (), method)
end
