using OrdinaryDiffEq
import KitAD as KA
import KitBase as KB

tspan = (0.0, 0.05)
dt = 0.001
tran = tspan[1]:dt*10:tspan[end]
ps = KB.PSpace1D(0, 1, 50, 0)
dx = ps.dx[1]
vs = KB.VSpace1D(-5, 5, 28)
u = vs.u
weights = vs.weights

prim0 = zeros(3, axes(ps.x, 1))
f0 = zeros(vs.nu, axes(ps.x, 1))
for i in axes(f0, 2)
    ρ = 1 + 0.1 * sin(2π * ps.x[i])
    λ = ρ
    prim0[:, i] .= [ρ, 1.0, λ]
    f0[:, i] .= KB.maxwellian(vs.u, prim0[:, i])
end

Kn0 = 1e-2
μ0 = KB.ref_vhs_vis(Kn0, 1.0, 0.5)
τ0 = μ0 * 2.0 * prim0[end]^(0.5) / prim0[1]
p0 = [μ0]

function rhs!(df, f, p, t)
    mu = abs(p[1])
    nu = size(f, 1)
    nx = size(f, 2)

    flux = zeros(nu, nx + 1)
    for j in 2:nx
        for i in 1:nu÷2
            flux[i, j] = u[i] * f[i, j]
        end
        for i in nu÷2+1:nu
            flux[i, j] = u[i] * f[i, j-1]
        end
    end
    for i in nu÷2+1:nu
        flux[i, 1] = u[i] * f[i, nx]
        flux[i, nx+1] = u[i] * f[i, nx]
    end
    for i in 1:nu÷2
        flux[i, 1] = u[i] * f[i, 1]
        flux[i, nx+1] = u[i] * f[i, 1]
    end

    w = reduce(hcat, map(1:nx) do i
        return KA.moments_conserve(f[:, i], u, weights)
    end)
    prim = reduce(hcat, map(1:nx) do i
        return KA.conserve_prim(w[:, i], 3)
    end)
    M = reduce(hcat, map(1:nx) do i
        return KA.maxwellian(u, prim[:, i])
    end)

    for j in axes(f, 2)
        tau = mu * 2.0 * prim[end, j]^0.5 / prim[1, j]
        for i in axes(f, 1)
            df[i, j] = (flux[i, j] - flux[i, j+1]) / dx + (M[i, j] - f[i, j]) / tau
        end
    end

    return nothing
end

p0 = [0.01]
prob0 = ODEProblem(rhs!, f0, tspan, p0)
sol0 = solve(prob0, Tsit5(); saveat=tran) |> Array

using EnsembleKalmanProcesses, Random
using EnsembleKalmanProcesses.ParameterDistributions: constrained_gaussian
using LinearAlgebra: I
using Statistics: mean
using KitBase.ProgressMeter: @showprogress

prob = ODEProblem(rhs!, f0, tspan, [10.0])
function loss(p)
    sol = solve(prob, Euler(); p = p, saveat = tran, dt=dt) |> Array

    loss = sum(abs2, sol .- sol0)
    return [loss]
end

loss(p0)

dim_output = 1
stabilization_level = 1e-2
Γ = stabilization_level * Matrix(I, dim_output, dim_output)
loss_target = [0.0]

prior = constrained_gaussian("u1", 0.0, 1e-1, -Inf, Inf)
N_ensemble = 10 * length(p0)
N_iterations = 50

rng = Random.seed!(Random.GLOBAL_RNG, 41)
initial_ensemble = construct_initial_ensemble(rng, prior, N_ensemble)
ensemble_kalman_process = EnsembleKalmanProcess(
    initial_ensemble,
    loss_target,
    Γ,
    Inversion(),
    scheduler = DefaultScheduler(1),
    accelerator = DefaultAccelerator(),
    localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    failure_handler_method = SampleSuccGauss(),
)

pe0 = get_u_final(ensemble_kalman_process)
loss([mean(pe0[iter, :]) for iter = 1:length(p0)]) |> first

@time while true
    params_i = get_u_final(ensemble_kalman_process)
    g_ens = zeros(1, N_ensemble)
    for i = 1:N_ensemble
        g = loss(params_i[:, i])[1]
        g_ens[1, i] = g
    end
    update_ensemble!(ensemble_kalman_process, g_ens)

    _p = get_u_final(ensemble_kalman_process)
    _l = loss([mean(_p[iter, :]) for iter = 1:length(p0)]) |> first
    println("loss: $_l")

    if _l < 1.1e-5
        break
    end
end

# 5e-5: 278.177291 seconds (14.30 G allocations: 300.477 GiB, 26.55% gc time, 0.01% compilation time)
# 1e-5: 455.609102 seconds (20.68 G allocations: 434.541 GiB, 28.62% gc time, 0.09% compilation time: 93% of which was recompilation)
