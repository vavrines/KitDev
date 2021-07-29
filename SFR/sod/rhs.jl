function dudt!(du, u, p, t)
    du .= 0.0
    J, ll, lr, lpdm, dgl, dgr, γ, uq = p

    nm = uq.nm
    nq = uq.nq

    ncell = size(u, 1)
    nsp = size(u, 2)

    u_ran = zeros(ncell, nsp, 3, nq)
    for i = 1:ncell
        for j = 1:nsp, k = 1:3
            u_ran[i, j, k, :] .= chaos_ran(u[i, j, k, :], uq)
        end

        tmp = @view u_ran[i, :, :, :]
        positive_limiter(tmp, γ, ps.wp ./ 2, uq.op.quad.weights, ll, lr, 0.99)
    end

    f = zeros(ncell, nsp, 3, nm+1)
    for i = 1:ncell, j = 1:nsp
        _f = zeros(3, nq)
        for k = 1:nq
            _f[:, k] .= euler_flux(u_ran[i, j, :, k], γ)[1] ./ J[i]
        end

        for k = 1:3
            f[i, j, k, :] .= ran_chaos(_f[k, :], uq)
        end
    end

    f_face = zeros(ncell, 3, nm+1, 2)
    for i = 1:ncell, j = 1:3, k = 1:nm+1
        f_face[i, j, k, 1] = dot(f[i, :, j, k], lr)
        f_face[i, j, k, 2] = dot(f[i, :, j, k], ll)
    end

    u_face = zeros(ncell, 3, nq, 2)
    for i = 1:ncell, j = 1:3, k = 1:nq
        u_face[i, j, k, 1] = dot(u_ran[i, :, j, k], lr)
        u_face[i, j, k, 2] = dot(u_ran[i, :, j, k], ll)
    end

    for i = 1:ncell, k = 1:nq
        primL = conserve_prim(u_face[i, :, k, 2], γ)
        primR = conserve_prim(u_face[i, :, k, 1], γ)
        if min(primL[end], primR[end]) < 0
            @info "negative interpolated temperature: ($(1/primL[end]), $(1/primR[end]))"

            #=u_mean = [sum(u_ran[i, :, idx, k] .* ps.wp ./ 2) for idx in axes(u_ran, 3)]
            for pp1 = 1:nsp
                u_ran[i, pp1, :, k] .= u_mean
            end=#
            tmp = @view u_ran[i, :, :, :]
            positive_limiter(tmp, γ, ps.wp ./ 2, uq.op.quad.weights, ll, lr, 0.9)

            for j = 1:3
                u_face[i, j, k, 1] = dot(u_ran[i, :, j, k], lr)
                u_face[i, j, k, 2] = dot(u_ran[i, :, j, k], ll)
            end
        end
    end

    fq_interaction = zeros(ncell + 1, 3, nq)
    for i = 2:ncell, j = 1:nq
        #=primL = conserve_prim(u_face[i-1, :, j, 1], γ)
        primR = conserve_prim(u_face[i, :, j, 1], γ)
        if primL[end] < 0
            primL[end] = 1 / 1e-6
            u_face[i-1, :, j, 1] .= prim_conserve(primL, γ)
        end
        if primR[end] < 0
            primR[end] = 1 / 1e-6
            u_face[i, :, j, 1] .= prim_conserve(primR, γ)
        end=#

        fw = @view fq_interaction[i, :, j]
        flux_hll!(fw, u_face[i-1, :, j, 1], u_face[i, :, j, 2], γ, 1.0)
    end

    f_interaction = zeros(ncell + 1, 3, nm+1)
    for i = 2:ncell, j = 1:3
        f_interaction[i, j, :] .= ran_chaos(fq_interaction[i, j, :], uq)
    end

    rhs1 = zero(u)
    for i = 1:ncell, ppp1 = 1:nsp, k = 1:3, l = 1:nm+1
        rhs1[i, ppp1, k, l] = dot(f[i, :, k, l], lpdm[ppp1, :])
    end

    idx = 2:ncell-1 # ending points are Dirichlet
    for i in idx, ppp1 = 1:nsp, k = 1:3, l = 1:nm+1
        du[i, ppp1, k, l] =
            -(
                rhs1[i, ppp1, k, l] +
                (f_interaction[i, k, l] / J[i] - f_face[i, k, l, 2]) * dgl[ppp1] +
                (f_interaction[i+1, k, l] / J[i] - f_face[i, k, l, 1]) * dgr[ppp1]
            )
    end
end
