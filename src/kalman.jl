"""
    StateSpaceModel(F, At, Ht, Q, R)

State-space representation of a dynamic system.

# Model description

State equation: ``ξ_{t+1} = F ξ_t + v_{t+1}``
Observation equation: ``y_t = A^T x_t + H^T ξ_t + w_t``

Error specification:
``E(v_t v_t^T) = Q``
``E(w_t w_t^T) = R``
``E(v_t v_τ^T) = E(w_t w_τ^T) = 0`` for ``t!=τ`` and ``E(v_t w_τ^T)=0``.

where:
- ``y_t`` is an ``(n×1)`` vector of observed variables;
- ``x_t`` is a ``(k×1)`` vector of exogenous observed variables;
- ``ξ_t`` is an ``(r×1)``, possibly unobserved, state vector.

Dimensions are inferred from the matrices passed to the constructor and are checked for consistency.

# Reference
Hamilton, J. (1994). Time Series Analysis.
"""
struct StateSpaceModel{MF<:AbstractMatrix, MA<:AbstractMatrix, MH<:AbstractMatrix,
                       MQ<:AbstractMatrix, MR<:AbstractMatrix
                      }
    # Seperate type parameter for each matrix to allow for static arrays of different sizes
    F::MF
    At::MA
    Ht::MH
    Q::MQ
    R::MR

    # Dimensions:
    n::Int
    k::Int
    r::Int

    function StateSpaceModel(F::MF, At::MA, Ht::MH, Q::MQ, R::MR) where {MF, MA, MH, MQ, MR}
        #F, At, Ht, Q, R = promote(F, At, Ht, Q, R)
        r = size(F, 1)
        n, k = size(At)

        size(F) == (r, r) || throw(ArgumentError("Expected size(F) == $((r,r))"))
        size(At) == (n, k) || throw(ArgumentError("Expected size(At) == $((n,k))"))
        size(Ht) == (n, r) || throw(ArgumentError("Expected size(Ht) == $((n,r))"))
        size(Q) == (r,r) || throw(ArgumentError("Expected size(Q) == $((r,r))"))
        size(R) == (n,n) || throw(ArgumentError("Expected size(R) == $((n,n))"))

        issymmetric(Q) || throw(ArgumentError("Q must be symmetric"))
        issymmetric(R) || throw(ArgumentError("R must be symmetric"))

        return new{MF, MA, MH, MQ, MR}(F, At, Ht, Q, R, n, k, r)
    end
end

"""
    KalmanPredictionIterator(ys, SSM::StateSpaceModel; xs, ξ1, P1)

Sequence of predictions ``ξ̂_{t+1|t}`` and associated mean squared error matrices ``P_{t+1|t}``, and conditional log-likelihoods
calculated by applying the Kalman filter.

Each element is a named tuple `(t, ξ, P, ll, y, x)`, where:
- `ξ` is ``ξ_{t|t-1}``;
- `P` is ``P_{t|t-1}``; 
- `ll` is the log-likelihood of ``y_t`` conditional on information up to `t`;
- and `(y,x)` is ``(y_t, x_t)``.

The log-likelihood is included as its calculation requires the same matrices as the Kalman updating step.
In addition to the assumptions documented for `StateSpaceModel`, the log-likelihood assumes that the errors
``v_t`` and ``w_t`` are normally distributed.

# Arguments
- `ys`: An iterator with elements ``y_t``.
- `SSM`: the state-space model from which to compute the filter.
- `xs`: An iterator with elements ``x_t``. Defaults to a constant.
- `ξ1`: ``E(ξ_1)``. Defaults to ``0``.
- `P1`: ``E([ξ_1-E(ξ_1)][ξ_1-E(ξ_1)]^T)``. Default is calculated assuming all eigenvalues of `SSM.F` are inside the unit circle.
"""
struct KalmanPredictionIterator{S<:StateSpaceModel, Mξ<:AbstractMatrix,
                                MP<:AbstractMatrix, T1, T2}
    SSM::S
    ξ1::Mξ # ξ_1|0
    P1::MP # P_1|0
    ys::T1
    xs::T2
end

KalmanPredictionIterator(ys, SSM::StateSpaceModel;
                         xs = repeat(ones(SSM.k), length(ys)),
                         ξ1 = zeros(SSM.r, 1),
                         P1 = reshape((I - kron(SSM.F, SSM.F)) \ vec(SSM.Q), SSM.r, SSM.r)
                        ) = KalmanPredictionIterator(SSM, ξ1, P1, ys, xs)

# Iteration interface: inherit size from ys
#Base.IteratorSize(::Type{KalmanPredictionIterator{M, Td}}) where {M, Td} = 
#    Base.IteratorSize(Td)
Base.length(KI::KalmanPredictionIterator) = length(KI.ys)
Base.size(KI::KalmanPredictionIterator, dim) = size(KI.ys, dim)
Base.size(KI::KalmanPredictionIterator) = size(KI.ys)
#Base.eltype(KI::KalmanPredictionIterator{M, Td}) where {M, Td} = (M, M, eltype(Td), eltype(Td))

# ===== Core iteration equations =====
function kalman_iteration_core(ξ, P, y::Number, x::Number, SSM)
    # Scalar y => no matrix inversion required
    (; F, Ht, R, Q, At) = SSM
    X = dot(vec(Ht), P, vec(Ht)) + R[1]
    fph = F * P * transpose(Ht)
    yerr = y - At[1] * x - dot(vec(Ht), vec(ξ))

    newP = F * P * transpose(F) - (fph * transpose(fph)) / X + Q
    newξ = F * ξ + fph * (yerr / X)
    ll = -0.5 * (log(2π) + yerr^2 / X + log(X))

    return newξ, newP, ll
end

function kalman_iteration_core(ξ, P, y, x, SSM)
    # General case
    (; F, Ht, R, Q, At, n) = SSM
    
    X = factorize(Symmetric(Ht * P * transpose(Ht) + R))
    fph = F * P * transpose(Ht)
    yerr = y .- At * x .- Ht * ξ

    newP = F * P * transpose(F) - fph * (X \ transpose(fph)) + Q
    newξ = F * ξ + fph * (X \ yerr)
    # log-likelihood is computed simultaneously as it uses the same matrices
    ll = -0.5 * (n * log(2π) + 
                 (transpose(yerr) * (X \ yerr))[1] +
                 logdet(X)
                )
    return newξ, Symmetric(newP), ll
end

function Base.iterate(KI::KalmanPredictionIterator)
    y_iter = iterate(KI.ys)
    isnothing(y_iter) && return nothing
    x_iter = iterate(KI.xs)
    isnothing(x_iter) && return nothing

    y1, y_state = y_iter
    x1, x_state = x_iter
    ξ1, P1 = KI.ξ1, KI.P1

    ξ2, P2, ll = kalman_iteration_core(ξ1, P1, y1, x1, KI.SSM)
    return (t=1, ξ=ξ1, P=P1, ll=ll, y=y1, x=x1), (2, ξ2, P2, y_state, x_state)
end

function Base.iterate(KI::KalmanPredictionIterator, state)
    t, ξt, Pt, y_state, x_state = state

    y_iter = iterate(KI.ys, y_state)
    isnothing(y_iter) && return nothing
    x_iter = iterate(KI.xs, x_state)
    isnothing(x_iter) && return nothing

    yt, y_state = y_iter
    xt, x_state = x_iter
    ξ_tplus1, P_tplus1, ll = kalman_iteration_core(ξt, Pt, yt, xt, KI.SSM)

    return (t=t, ξ=ξt, P=Pt, ll=ll, y=yt, x=xt), (t+1, ξ_tplus1, P_tplus1, y_state, x_state)
end

"""
    kalman_log_likelihood(iter::KalmanPredictionIterator)

Compute the sample log-likelihood for `iter`.

Assumes that the errors ``v_t`` and ``w_t`` are both normally distributed.
"""
function kalman_log_likelihood(iter::KalmanPredictionIterator)
    return mapreduce(el -> el.ll, +, iter)
end
