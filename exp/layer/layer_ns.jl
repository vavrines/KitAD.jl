using OrdinaryDiffEq, KitBase
using Base.Threads: @threads

function flux_opt!(fw, wL, wR, inK, γ, μᵣ, ω, dt, dxL, dxR)
    primL = conserve_prim(wL, γ)
    primR = conserve_prim(wR, γ)

    Mu1, Mv1, Mw1, MuL1, MuR1 = KB.gauss_moments(primL, inK)
    Mu2, Mv2, Mw2, MuL2, MuR2 = KB.gauss_moments(primR, inK)

    w =
        primL[1] .* moments_conserve(MuL1, Mv1, Mw1, 0, 0, 0) .+
        primR[1] .* moments_conserve(MuR2, Mv2, Mw2, 0, 0, 0)
    prim = conserve_prim(w, γ)
    tau =
        vhs_collision_time(prim, μᵣ, ω) +
        2.0 * dt * abs(primL[1] / primL[end] - primR[1] / primR[end]) /
        (primL[1] / primL[end] + primR[1] / primR[end])

    Mu, Mv, Mw, MuL, MuR = KB.gauss_moments(prim, inK)
    sw0L = (w .- (wL)) ./ dxL
    sw0R = ((wR) .- w) ./ dxR
    gaL = KB.pdf_slope(prim, sw0L, inK)
    gaR = KB.pdf_slope(prim, sw0R, inK)
    sw =
        -prim[1] .* (KB.moments_conserve_slope(gaL, MuL, Mv, Mw, 1, 0, 0) .+
         KB.moments_conserve_slope(gaR, MuR, Mv, Mw, 1, 0, 0))
    gaT = KB.pdf_slope(prim, sw, inK)

    # time-integration constants
    Mt = zeros(5)
    Mt[4] = tau * (1.0 - exp(-dt / tau))
    Mt[5] = -tau * dt * exp(-dt / tau) + tau * Mt[4]
    Mt[1] = dt - Mt[4]
    Mt[2] = -tau * Mt[1] + Mt[5]
    Mt[3] = 0.5 * dt^2 - tau * Mt[1]

    # central flux
    Muv = moments_conserve(Mu, Mv, Mw, 1, 0, 0)
    MauL = KB.moments_conserve_slope(gaL, MuL, Mv, Mw, 2, 0, 0)
    MauR = KB.moments_conserve_slope(gaR, MuR, Mv, Mw, 2, 0, 0)
    MauT = KB.moments_conserve_slope(gaT, Mu, Mv, Mw, 1, 0, 0)

    fw .=
        Mt[1] .* prim[1] .* Muv .+ Mt[2] .* prim[1] .* (MauL .+ MauR) .+
        Mt[3] .* prim[1] .* MauT

    # upwind flux
    MuvL = moments_conserve(MuL1, Mv1, Mw1, 1, 0, 0)
    MuvR = moments_conserve(MuR2, Mv2, Mw2, 1, 0, 0)

    @. fw += Mt[4] * primL[1] * MuvL + Mt[4] * primR[1] * MuvR
    @. fw /= dt

    return nothing
end

ps = PSpace1D(-0.1, 0.1, 80, 0)
gas = Gas(; Kn=5e-3, K=1.0)

w0 = zeros(4, ps.nx)
prim0 = zeros(4, ps.nx)
for i in 1:ps.nx
    if ps.x[i] <= 0
        prim0[:, i] .= [1.0, 0.0, 1.0, 1.0]
    else
        prim0[:, i] .= [1.0, 0.0, -1.0, 2.0]
    end

    w0[:, i] .= prim_conserve(prim0[:, i], gas.γ)
end

τ0 = vhs_collision_time(prim0[:, 1], gas.μᵣ, gas.ω)
tmax = 10τ0
tspan = (0.0, tmax)
dt = tmax / 20 / 10
tran = tspan[1]:tmax/20:tspan[end]

function rhs!(dw, w, p, t)
    ps, gas = p
    nx = size(w, 2)

    flux = zeros(4, nx + 1)
    @inbounds @threads for j in 2:nx
        fw = @view flux[:, j]
        flux_opt!(
            fw,
            w[:, j-1],
            w[:, j],
            gas.K,
            gas.γ,
            gas.μᵣ,
            gas.ω,
            dt,
            0.5 * ps.dx[j-1],
            0.5 * ps.dx[j],
        )
    end

    @inbounds @threads for j in 2:nx-1
        for i in 1:4
            dw[i, j] = (flux[i, j] - flux[i, j+1]) / ps.dx[j]
        end
    end

    return nothing
end

p = (ps, gas)
prob = ODEProblem(rhs!, w0, tspan, p)
res = solve(prob, Tsit5(); saveat=tran)
resArr = Array(res)

# field
sol = zeros(ps.nx, 4)
for i in axes(sol, 1)
    w = res.u[end][:, i]
    prim = conserve_prim(w, gas.γ)
    sol[i, :] .= prim
    sol[i, end] = 1 / sol[i, end]
end

using Plots
plot(ps.x[1:ps.nx], sol[:, 1])

using KitBase.JLD2
cd(@__DIR__)
@save "layer_nssol.jld2" resArr
