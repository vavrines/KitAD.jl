"""
Optimization of numerical flux function for Euler equations

"""

using OrdinaryDiffEq, SciMLSensitivity, Solaris
using Flux: sigmoid
import KitAD as KA
import KitBase as KB

const γ = 5 / 3

function flux_opt!(fw, wL, wR, p)
    Mt = zeros(2)
    Mt[2] = sigmoid(p)
    Mt[1] = 1.0 - Mt[2]

    primL = KB.conserve_prim(wL, γ)
    primR = KB.conserve_prim(wR, γ)
    Mu1, Mxi1, MuL1, MuR1 = KB.gauss_moments(primL, 2)
    Mu2, Mxi2, MuL2, MuR2 = KB.gauss_moments(primR, 2)

    w =
        primL[1] .* KB.moments_conserve(MuL1, Mxi1, 0, 0) .+
        primR[1] .* KB.moments_conserve(MuR2, Mxi2, 0, 0)
    prim = KB.conserve_prim(w, γ)
    Mu, Mxi, MuL, MuR = KB.gauss_moments(prim, 2)

    # central
    Muv = KB.moments_conserve(Mu, Mxi, 1, 0)
    fw .= Mt[1] .* prim[1] .* Muv

    # upwind
    MuvL = KB.moments_conserve(MuL1, Mxi1, 1, 0)
    MuvR = KB.moments_conserve(MuR2, Mxi2, 1, 0)

    @. fw += Mt[2] * primL[1] * MuvL + Mt[2] * primR[1] * MuvR

    return nothing
end

tspan = (0.0, 0.2)
dt = 0.001
tran = tspan[1]:dt*10:tspan[end]
ps = KB.PSpace1D(0, 1, 200)
dx = ps.dx[1]

begin
    _s = KB.sample_riemann_solution(
        ps.x[1:ps.nx] .- 0.5,
        tspan[end],
        KB.HydroStatus(1.0, 0.0, 1.0, γ),
        KB.HydroStatus(0.125, 0.0, 0.1, γ),
        KB.PrimitiveValue(),
        1e-12,
    )
    sole_prim = zeros(3, ps.nx)
    sole_cons = zeros(3, ps.nx)
    for i in 1:ps.nx
        sole_prim[1, i] = _s[i].rho
        sole_prim[2, i] = _s[i].u
        sole_prim[3, i] = _s[i].rho / (_s[i].p * 2)

        sole_cons[:, i] .= KB.prim_conserve(sole_prim[:, i], γ)
    end
end

prim0 = zeros(3, axes(ps.x, 1))
w0 = zeros(3, axes(ps.x, 1))
for i in axes(w0, 2)
    if ps.x[i] < 0.5
        prim0[:, i] .= [1.0, 0.0, 0.5]
    else
        prim0[:, i] .= [0.125, 0.0, 0.625]
    end

    w0[:, i] .= KB.prim_conserve(prim0[:, i], γ)
end

function rhs!(dw, w, p, t)
    nx = size(w, 2)

    flux = zeros(3, nx + 1)
    @inbounds for j in 2:nx
        fw = @view flux[:, j]
        wL = @view w[:, j-1]
        wR = @view w[:, j]
        #flux_opt!(fw, wL, wR, p[1])
        flux_opt!(fw, wL, wR, p[j])
    end

    @inbounds for j in 2:nx-1
        for i in axes(w, 1)
            dw[i, j] = (flux[i, j] - flux[i, j+1]) / dx
        end
    end

    return nothing
end

p0 = ones(Float64, ps.nx + 1) .* 5
prob0 = ODEProblem(rhs!, w0, tspan, p0)
sol0 = solve(prob0, Tsit5(); saveat=tspan[2]) |> Array

function loss(p)
    prob = ODEProblem(rhs!, w0, tspan, p)
    sol = solve(prob, Euler(); saveat=tspan[end], dt=dt)
    l = sum(abs2, sol.u[end] .- sole_cons)

    return l
end

loss(p0)

cb = function (p, l)
    println("loss: $(l)")
    return false
end

res = sci_train(loss, p0, Adam(0.05); cb=cb, iters=100, ad=AutoZygote())
res = sci_train(loss, res.u, AdamW(0.01); cb=cb, iters=100, ad=AutoZygote())
#
# ~ 0.30x
#

prob1 = ODEProblem(rhs!, w0, tspan, res.u)
sol1 = solve(prob1, Tsit5(); saveat=tspan[2]) |> Array

begin
    using Plots
    solcons0 = zeros(ps.nx, 3)
    solcons1 = zeros(ps.nx, 3)
    solprim0 = zeros(ps.nx, 3)
    solprim1 = zeros(ps.nx, 3)
    for i in axes(solprim0, 1)
        solcons0[i, :] .= sol0[:, i, end]
        solcons1[i, :] .= sol1[:, i, end]
        solprim0[i, :] .= KB.conserve_prim(sol0[:, i, end], γ)
        solprim1[i, :] .= KB.conserve_prim(sol1[:, i, end], γ)
    end
end

begin
    idx = 1
    plot(ps.x[1:ps.nx], solprim0[:, idx]; label="original")
    plot!(ps.x[1:ps.nx], solprim1[:, idx]; label="optimized")
    plot!(ps.x[1:ps.nx], sole_prim[idx, :]; label="exact")
end

plot(ps.x[2:ps.nx], res.u[2:ps.nx])

using KitBase.JLD2
cd(@__DIR__)
u = res.u
@save "sodpara.jld2" u # ~ -1.9244146828496453
