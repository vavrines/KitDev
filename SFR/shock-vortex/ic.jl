"""
Calculate initial vortex condition

κ: vortex strength
μ: decay rate
rc: critical radius

"""
function vortex_ic!(
    prim,
    γ,
    x,
    y,
    κ = 0.25,
    μ = 0.204,
    rc = 0.05;
    x0 = 0.8,
    y0 = 0.5,
)
    T0 = 1 / prim[end]
    s = prim[1]^(1-γ) / (2 * prim[end])

    r = sqrt((x - x0)^2 + (y - y0)^2)    
    η = r / rc
    
    δu = κ * η * exp(μ * (1-η^2)) * (y - y0) / r
    δv = -κ * η * exp(μ * (1-η^2)) * (x - x0) / r
    δT = -(γ-1)*κ^2/(8*μ*γ)*exp(2*μ*(1-η^2))

    ρ = prim[1]^(γ-1) * (T0+δT) / T0^(1/(γ-1))
    prim .= [ρ, prim[2]+δu, prim[3]+δv, 1/(1/prim[4]+δT)]

    return nothing
end