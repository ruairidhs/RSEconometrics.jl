# Description:
# Some convenience functions for maximum likelihood

# Assume we have nll(u, p)
function _second_derivative_I(nll, sol, p)
    # returns T × I_{2D}
    return ForwardDiff.hessian(u -> nll(u, p), sol)
end

function _outer_product_I(nlls, sol, p)
    # now nlls returns [ll_1, …, ll_T]
    J = ForwardDiff.jacobian(u -> nlls(u, p), sol)
    return transpose(J) * J
end

function _sandwich_I(nlls, sol, p)
    I2D = _second_derivative_I((u, p) -> sum(nlls(u, p)), sol, p)
    IOP = _outer_product_I(nlls, sol, p)
    return I2D * (IOP \ I2D)
end

function ml_varcov(nll, sol, p, method)
    if method == :hessian
        return inv(_second_derivative_I(nll, sol, p))
    elseif method == :outer_product
        return inv(_outer_product_I(nll, sol, p))
    elseif method == :sandwich
        return inv(_sandwich_I(nll, sol, p))
    else
        throw(ArgumentError("Unknown method. Available methods: [:hessian, :outer_product, :sandwich]"))
    end
end

get_std_errs(varcov) = sqrt.(diag(varcov))
