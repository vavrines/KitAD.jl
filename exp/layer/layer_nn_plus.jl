"""
Machine learning assisted modeling for hydrodynamic closure

"""

using OrdinaryDiffEq, KitBase, SciMLSensitivity, Solaris
using KitBase.JLD2
using Solaris.Flux: sigmoid
import KitAD as KA

###
newrun = true
#newrun = false
###

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

nn = FnChain(FnDense(8, 32, tanh), FnDense(32, 4, tanh))
p0 = init_params(nn) #.|> Float64

ps = PSpace1D(-0.1, 0.1, 80, 0)
gas = Gas(; Kn=5e-3, K=1.0)

w0 = zeros(Float32, 4, ps.nx)
prim0 = zeros(Float32, 4, ps.nx)
for i in 1:ps.nx
    if ps.x[i] <= 0
        prim0[:, i] .= [1.0, 0.0, 1.0, 1.0]
    else
        prim0[:, i] .= [1.0, 0.0, -1.0, 2.0]
    end

    w0[:, i] .= prim_conserve(prim0[:, i], gas.γ)
end

begin
    τ0 = vhs_collision_time(prim0[:, 1], gas.μᵣ, gas.ω)
    tmax = 10τ0
    tspan = (0.0, tmax)
    dt = tmax / 20 / 10
    tran = tspan[1]:dt*10:tspan[end]
    dx = ps.dx[1]
    K = gas.K
    γ = gas.γ
    μᵣ = gas.μᵣ
    ω = gas.ω
end

function rhs0!(dw, w, p, t)
    nx = size(w, 2)

    flux = zeros(Float32, 4, nx + 1)
    for j in 2:nx
        flux[:, j] .= flux_basis(w[:, j-1], w[:, j])
    end

    for j in 2:nx-1
        for i in 1:4
            dw[i, j] = (flux[i, j] - flux[i, j+1]) / dx
        end
    end

    return nothing
end
function rhs!(dw, w, p, t)
    nx = size(w, 2)

    flux = zeros(Float32, 4, nx + 1)
    for j in 2:nx
        flux[:, j] .= flux_basis(w[:, j-1], w[:, j])
        flux[:, j] .+= nn(vcat(w[:, j-1], w[:, j]), p) .* 0.1
    end

    for j in 2:nx-1
        for i in 1:4
            dw[i, j] = (flux[i, j] - flux[i, j+1]) / dx
            dw[i, 1] = 0.0
            dw[i, nx] = 0.0
        end
    end

    return nothing
end

function loss(p)
    prob = ODEProblem(rhs!, w0, tspan, p)
    sol = solve(prob, Euler(); saveat=tran, dt=dt) |> Array
    l = sum(abs2, sol .- resArr)
    l += sum(abs2, p) * 1e-5
    l += loss_tv(sol) * 1e-3

    return l
end

function loss_tv(sol)
    dx = size(sol, 2)
    dt = size(sol, 3)
    tv = 0.0f0
    for i in 1:dt
        for j in 1:dx-1
            tv += abs(sol[1, j+1, i] - sol[1, j, i])
        end
    end

    return tv
end

cb = function (p, l)
    println("loss: $(l)")
    return false
end

if newrun
    u = deepcopy(p0)
else
    @load "layer_param.jld2" u
end

M = loss(u)

GC.@preserve res = sci_train(loss, u, Adam(0.05); cb=cb, iters=30, ad=AutoZygote())
GC.@preserve res = sci_train(loss, res.u, LBFGS(); cb=cb, iters=20, ad=AutoZygote())
GC.@preserve res = sci_train(loss, res.u, AdamW(0.001); cb=cb, iters=100, ad=AutoZygote())

# ~0.129

u = deepcopy(res.u)
@save "layer_param.jld2" u
