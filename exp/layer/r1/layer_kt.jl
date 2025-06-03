using OrdinaryDiffEq, KitBase
using Base.Threads: @threads

function df!(dh, db, h, b, fhL, fbL, fhR, fbR, u, v, weights, K, γ, μᵣ, ω, Δs)
    w = moments_conserve(h, b, u, v, weights, KB.VDF{2,2})
    prim = conserve_prim(w, γ)

    MH = maxwellian(u, v, prim)
    MB = MH .* K ./ (2.0 * prim[end])
    τ = vhs_collision_time(prim, μᵣ, ω)

    for i in eachindex(u)
        dh[i] = (fhL[i] - fhR[i]) / Δs + (MH[i] - h[i]) / τ
        db[i] = (fbL[i] - fbR[i]) / Δs + (MB[i] - b[i]) / τ
    end

    return nothing
end

ps = PSpace1D(-0.1, 0.1, 80, 0)
vs = VSpace2D(-5, 5, 28, -5, 5, 64)
gas = Gas(; Kn=5e-3, K=1.0)

w0 = zeros(4, ps.nx)
prim0 = zeros(4, ps.nx)
f0 = zeros(vs.nu, vs.nv, 2, ps.nx)
for i in 1:ps.nx
    if ps.x[i] <= 0
        prim0[:, i] .= [1.0, 0.0, 0.5, 1.0]
    else
        prim0[:, i] .= [1.0, 0.0, -1.0, 5 / 3]
    end

    w0[:, i] .= prim_conserve(prim0[:, i], gas.γ)
    f0[:, :, 1, i] .= maxwellian(vs.u, vs.v, prim0[:, i])
    f0[:, :, 2, i] .= energy_maxwellian(f0[:, :, 1, i], prim0[:, i], gas.K)
end

τ0 = vhs_collision_time(prim0[:, 1], gas.μᵣ, gas.ω)
tmax = 10τ0
#tmax = 15τ0
tspan = (0.0, tmax)
tran = tspan[1]:tmax/20:tspan[end]

function rhs!(df, f, p, t)
    ps, vs, gas = p

    nu = size(f, 1)
    nv = size(f, 2)
    nx = size(f, 4)

    flux = zeros(nu, nv, 2, nx + 1)
    @inbounds @threads for j in 2:nx
        fh = @view flux[:, :, 1, j]
        fb = @view flux[:, :, 2, j]
        flux_kfvs!(
            zeros(4),
            fh,
            fb,
            f[:, :, 1, j-1],
            f[:, :, 2, j-1],
            f[:, :, 1, j],
            f[:, :, 2, j],
            vs.u,
            vs.v,
            vs.weights,
            1.0,
            1.0,
        )
    end

    @inbounds @threads for j in 2:nx-1
        dh = @view df[:, :, 1, j]
        db = @view df[:, :, 2, j]
        df!(
            dh,
            db,
            f[:, :, 1, j],
            f[:, :, 2, j],
            flux[:, :, 1, j],
            flux[:, :, 2, j],
            flux[:, :, 1, j+1],
            flux[:, :, 2, j+1],
            vs.u,
            vs.v,
            vs.weights,
            gas.K,
            gas.γ,
            gas.μᵣ,
            gas.ω,
            ps.dx[j],
        )
    end

    return nothing
end

p = (ps, vs, gas)
prob = ODEProblem(rhs!, f0, tspan, p)
res = solve(prob, Midpoint(); saveat=tran)

resArr = zeros(4, ps.nx, length(tran))
for j in 1:length(tran)
    for i in 1:ps.nx
        f = res.u[j]
        w = moments_conserve(
            f[:, :, 1, i],
            f[:, :, 2, i],
            vs.u,
            vs.v,
            vs.weights,
            KB.VDF{2,2},
        )
        resArr[:, i, j] .= w
    end
end

# field
sol = zeros(ps.nx, 4)
for i in axes(sol, 1)
    f = res.u[end]
    w = moments_conserve(f[:, :, 1, i], f[:, :, 2, i], vs.u, vs.v, vs.weights, KB.VDF{2,2})
    prim = conserve_prim(w, gas.γ)
    sol[i, :] .= prim
    sol[i, end] = 1 / sol[i, end]
end

using Plots
plot(ps.x[1:ps.nx], sol[:, 1])

using KitBase.JLD2
cd(@__DIR__)
@save "layer_ktsol.jld2" resArr
