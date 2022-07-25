function regime_data(ks, w, prim, swx, swy, f)
    Mu, Mv, Mxi, _, _1 = gauss_moments(prim, ks.gas.K)
    a = pdf_slope(prim, swx, ks.gas.K)
    b = pdf_slope(prim, swy, ks.gas.K)
    sw =
        -prim[1] .* (
            moments_conserve_slope(a, Mu, Mv, Mxi, 1, 0) .+
            moments_conserve_slope(b, Mu, Mv, Mxi, 0, 1)
        )
    A = pdf_slope(prim, sw, ks.gas.K)
    tau = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)

    fr = chapman_enskog(ks.vs.u, ks.vs.v, prim, a, b, A, tau)
    L = norm((f .- fr) ./ prim[1])

    sw = (swx .^ 2 + swy .^ 2) .^ 0.5
    x = [w; sw; tau]
    y = ifelse(L <= 0.005, [1.0, 0.0], [0.0, 1.0])

    return x, y
end
