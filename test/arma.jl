# Tests:
#   - autocovariance functions are roughly satisfied
@testset verbose=true "ARMA" begin
    @testset "MA(1)" begin
        θ, μ, σ2 = 0.3, 1.0, 0.5 
        ma_spec = construct_ma([θ]; μ=μ, σ2=σ2) 
        data = rand(ma_spec, 10_000; burn_in=100)
        acv = SB.autocov(data, [0, 1, 2])
        @test isapprox(acv[1], (1 + θ^2) * σ2; rtol=0.2)
        @test isapprox(acv[2], θ * σ2; rtol=0.2)
        @test abs(acv[3]) <= 1e-2
    end   
    
    @testset "AR(1)" begin
        ϕ, μ, σ2 = 0.3, 1.0, 0.5
        ar_spec = construct_ar([ϕ]; μ=μ, σ2=σ2)
        data = rand(ar_spec, 10_000; burn_in=100)
        acv = SB.autocov(data, 0:3)
        for i in 0:3
            @test isapprox(acv[i+1], σ2 * ϕ^i / (1-ϕ^2); rtol=0.2)
        end
    end
end
