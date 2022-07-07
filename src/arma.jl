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
    StateSpaceModel(arma::UnivariateARMA)

Construct the state-space representation of `arma`.
"""
function StateSpaceModel(arma::UnivariateARMA)
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
