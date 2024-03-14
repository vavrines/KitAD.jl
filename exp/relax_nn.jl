"""
Machine learning assisted modeling for kinetic collision operator

"""

using OrdinaryDiffEq, SciMLSensitivity, Solaris, KitBase
using Optimization: AutoZygote
using Optimisers: Adam
using Optim: LBFGS
using BenchmarkTools, Plots

set = config_ntuple(
    u0 = -5,
    u1 = 5,
    nu = 80,
    v0 = -5,
    v1 = 5,
    nv = 28,
    w0 = -5,
    w1 = 5,
    nw = 28,
    t1 = 8,
    nt = 81,
    Kn = 1,
)

tspan = (0, set.t1)
tsteps = linspace(tspan[1], tspan[2], set.nt)
γ = 5 / 3

vs = VSpace1D(set.u0, set.u1, set.nu)
vs2 = VSpace2D(set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)
vs3 = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)

f0 = 0.5 * (1 / π)^1.5 .*
    (exp.(-(vs3.u .- 1) .^ 2) .+ 0.7 .* exp.(-(vs3.u .+ 1) .^ 2)) .*
    exp.(-vs3.v .^ 2) .* exp.(-vs3.w .^ 2)

mu_ref = ref_vhs_vis(set.Kn, set.α, set.ω)
kn_bzm = hs_boltz_kn(mu_ref, 1.0)

w0 = moments_conserve(f0, vs3.u, vs3.v, vs3.w, vs3.weights)
prim0 = conserve_prim(w0, γ)
M0 = maxwellian(vs3.u, vs3.v, vs3.w, prim0)
τ0 = mu_ref * 2.0 * prim0[end]^(0.5) / prim0[1]

phi, psi, chi = kernel_mode(
    set.nm,
    vs3.u1,
    vs3.v1,
    vs3.w1,
    vs3.du[1, 1, 1],
    vs3.dv[1, 1, 1],
    vs3.dw[1, 1, 1],
    vs3.nu,
    vs3.nv,
    vs3.nw,
    set.α,
)
prob = ODEProblem(boltzmann_ode!, f0, tspan, [kn_bzm, set.nm, phi, psi, chi])
data_boltz = solve(prob, Tsit5(), saveat = tsteps) |> Array

#@benchmark solve(prob, Tsit5(), saveat = tsteps) |> Array
"""
BenchmarkTools.Trial: 3 samples with 1 evaluation.
 Range (min … max):  1.980 s …   2.015 s  ┊ GC (min … max): 2.00% … 2.03%
 Time  (median):     1.997 s              ┊ GC (median):    2.05%
 Time  (mean ± σ):   1.997 s ± 17.285 ms  ┊ GC (mean ± σ):  2.23% ± 0.38%

  █                          █                            █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  1.98 s         Histogram: frequency by time        2.01 s <

 Memory estimate: 6.42 GiB, allocs estimate: 28241.
"""

prob1 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ0])
data_bgk = solve(prob1, Tsit5(), saveat = tsteps) |> Array

#@benchmark solve(prob1, Tsit5(), saveat = tsteps) |> Array
"""
BenchmarkTools.Trial: 145 samples with 1 evaluation.
 Range (min … max):  28.085 ms … 45.414 ms  ┊ GC (min … max): 11.29% … 13.63%
 Time  (median):     34.111 ms              ┊ GC (median):     9.90%
 Time  (mean ± σ):   34.543 ms ±  4.301 ms  ┊ GC (mean ± σ):   9.37% ±  7.52%

  ▃▁        ▂▄         ▇              █                        
  ██▃▅▃▁▁▁▁▃██▆▇▅▃▃▃▄▅▆██▃▄▃▃▁▁▄▁▃▁▁▃▃█▆▅▄▁▃▁▇▆▆▄▅▁▃▃▁▁▁▁▁▁▁▃ ▃
  28.1 ms         Histogram: frequency by time        44.6 ms <

 Memory estimate: 124.44 MiB, allocs estimate: 581.
"""

data_boltz_1D = zeros(axes(data_boltz, 1), axes(data_boltz, 4))
data_bgk_1D = zeros(axes(data_bgk, 1), axes(data_bgk, 4))
for j in axes(data_boltz_1D, 2)
    data_boltz_1D[:, j] .=
        reduce_distribution(data_boltz[:, :, :, j], vs2.weights)
    data_bgk_1D[:, j] .=
        reduce_distribution(data_bgk[:, :, :, j], vs2.weights)
end
h0_1D, b0_1D = reduce_distribution(f0, vs3.v, vs3.w, vs2.weights)
H0_1D, B0_1D = reduce_distribution(M0, vs3.v, vs3.w, vs2.weights)

nn = FnChain(FnDense(vs.nu, vs.nu*4, tanh), FnDense(vs.nu*4, vs.nu))
p0 = init_params(nn)

#=function rhs!(df, f, p, t)
    #df .= nn(f, p)
    df .= (H0_1D .- f) ./ τ0 .+ nn(H0_1D .- f, p)

    return nothing
en=#

function rhs(f, p, t)
    return (H0_1D .- f) ./ τ0 .+ nn(H0_1D .- f, p)
end

function loss(p)
    ube = ODEProblem(rhs, h0_1D, tspan, p)
    pred = solve(ube, Midpoint(), saveat = tsteps) |> Array
    loss = sum(abs2, data_boltz_1D .- pred)

    return loss
end

#his = []
cb = function (p, l)
    println("loss: $(l)")
    #push!(his, l)
    return false
end

res = sci_train(loss, p0, Adam(0.05); cb = cb, iters = 100, ad = AutoZygote())
res = sci_train(loss, res.u, LBFGS(); cb = cb, iters = 100, ad = AutoZygote())
res = sci_train(loss, res.u, Adam(0.01); cb = cb, iters = 100, ad = AutoZygote())

ube = ODEProblem(rhs, h0_1D, tspan, p0)
data_nn = solve(ube, Tsit5(), u0 = h0_1D, p = res.u, saveat = tsteps) |> Array

#@benchmark solve(ube, Tsit5(), saveat = tsteps) |> Array
"""
BenchmarkTools.Trial: 825 samples with 1 evaluation.
 Range (min … max):  5.367 ms …  11.696 ms  ┊ GC (min … max): 0.00% … 30.45%
 Time  (median):     5.624 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   6.059 ms ± 619.857 μs  ┊ GC (mean ± σ):  7.97% ±  7.85%

    ▂   ▆█▅                                    ▁▂▃▃▁▁          
  ▆▅██▆█████▃▄▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▂▁▁▁▁▁▁▁▂▁▁▁▁▁▂▃▃▅███████▆▅▃▃▃▂▂ ▃
  5.37 ms         Histogram: frequency by time        6.88 ms <

 Memory estimate: 60.27 MiB, allocs estimate: 6109.
"""

idx = 21
plot(
    vs.u,
    data_boltz_1D[:, 1],
    lw = 1.5,
    label = "initial",
    color = :gray32,
    xlabel = "u",
    ylabel = "particle distribution",
)
plot!(
    vs.u,
    data_boltz_1D[:, idx],
    lw = 1.5,
    label = "Boltzmann",
    color = 1,
)
plot!(
    vs.u,
    data_bgk_1D[:, idx],
    lw = 1.5,
    line = :dash,
    label = "BGK",
    color = 2,
)
plot!(
    vs.u,
    H0_1D,
    lw = 1.5,
    label = "Maxwellian",
    color = 10,
)
scatter!(vs.u, data_nn[:, idx], lw = 1.5, label = "neural", color = 3, alpha = 0.7)

using KitBase.JLD2
cd(@__DIR__)
u = res.u
@save "relax_params.jld2" u
