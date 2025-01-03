"""
Machine learning assisted modeling for hydrodynamic closure

"""

using OrdinaryDiffEq, KitBase, SciMLSensitivity, Solaris
using KitBase.JLD2
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

@load "layer_param.jld2" u

prob = ODEProblem(rhs!, w0, tspan, u)
sol = solve(prob, Euler(); saveat=tran, dt=dt) |> Array
sol = solve(prob, Tsit5(); saveat=tran) |> Array

prob0 = ODEProblem(rhs0!, w0, tspan, u)
sol0 = solve(prob0, Euler(); saveat=tran, dt=dt) |> Array
sol0 = solve(prob0, Tsit5(); saveat=tran) |> Array

let idx = 4, itx = 21
    plot(ps.x, sol[idx, :, itx])
    plot!(ps.x, sol0[idx, :, itx])
    plot!(ps.x, resArr[idx, :, itx])
end

idx = 4
plot(ps.x, 1 ./ sol[idx, :, end])
plot!(ps.x, 1 ./ sol0[idx, :, end])
plot!(ps.x, 1 ./ resArr[idx, :, end])
