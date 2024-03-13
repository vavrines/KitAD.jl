"""
Machine learning assisted modeling for hydrodynamic closure

"""

using OrdinaryDiffEq, KitBase, SciMLSensitivity, Solaris
using KitBase.JLD2
using Optimization: AutoZygote
using Optimisers: Adam
using Optim: LBFGS
using Base.Threads: @threads
using Plots
using Solaris.Flux: sigmoid
import KitAD as KA

function flux_basis(wL, wR)
    primL = conserve_prim(wL, γ)
    primR = conserve_prim(wR, γ)
    Mu1, Mv1, Mw1, MuL1, MuR1 = KB.gauss_moments(primL, 1)
    Mu2, Mv2, Mw2, MuL2, MuR2 = KB.gauss_moments(primR, 1)
    MuvL = moments_conserve(MuL1, Mv1, Mw1, 1, 0, 0)
    MuvR = moments_conserve(MuR2, Mv2, Mw2, 1, 0, 0)
    fw = primL[1] * MuvL + primR[1] * MuvR

    return fw
end

cd(@__DIR__)
@load "layer_ktsol.jld2" resArr

nn = FnChain(FnDense(8, 24, tanh), FnDense(24, 4, sigmoid))
#nn = FnChain(FnDense(8, 24, tanh), FnDense(24, 4))
p0 = init_params(nn) #.|> Float64

ps = PSpace1D(-0.5, 0.5, 500, 0)
gas = Gas(Kn = 5e-3, K = 1.0)

w0 = zeros(4, ps.nx)
prim0 = zeros(4, ps.nx)
for i = 1:ps.nx
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
dt = 0.5 * ps.dx[1] / 5
tran = tspan[1]:dt*10:tspan[end]
dx = ps.dx[1]
K = gas.K
γ = gas.γ
μᵣ = gas.μᵣ
ω = gas.ω

function rhs0!(dw, w, p, t)
    nx = size(w, 2)

    flux = zeros(4, nx+1)
    for j = 2:nx
        flux[:, j] .= flux_basis(w[:, j-1], w[:, j])
    end

    for j = 2:nx-1
        for i = 1:4
            dw[i, j] = (flux[i, j] - flux[i, j+1]) / dx
        end
    end

    return nothing
end
function rhs!(dw, w, p, t)
    nx = size(w, 2)

    flux = zeros(4, nx+1)
    for j = 2:nx
        flux[:, j] .= flux_basis(w[:, j-1], w[:, j])
        flux[:, j] .*= nn(vcat(w[:, j-1], w[:, j]), p)
    end

    for j = 2:nx-1
        for i = 1:4
            dw[i, j] = (flux[i, j] - flux[i, j+1]) / dx
        end
    end

    return nothing
end

function loss(p)
    prob = ODEProblem(rhs!, w0, tspan, p)
    sol = solve(prob, Euler(), saveat = tran, dt = dt) |> Array
    l = sum(abs2, sol .- resArr)

    return l
end

loss(p0)

cb = function (p, l)
    println("loss: $(l)")
    return false
end

res = sci_train(loss, p0, Adam(0.05); cb = cb, iters = 50, ad = AutoZygote())
res = sci_train(loss, res.u, Adam(0.05); cb = cb, iters = 50, ad = AutoZygote())


prob0 = ODEProblem(rhs0!, w0, tspan, res.u)
sol0 = solve(prob0, Euler(), saveat = tran, dt = dt) |> Array

prob = ODEProblem(rhs!, w0, tspan, res.u)
sol = solve(prob, Euler(), saveat = tran, dt = dt) |> Array

idx = 4
plot(ps.x, 1 ./ sol[idx, :, end])
plot!(ps.x, 1 ./ sol0[idx, :, end])
plot!(ps.x, 1 ./ resArr[idx, :, end])


u = res.u
@save "layer_param.jld2" u
