using OrdinaryDiffEq, SciMLSensitivity, Solaris, KitBase
using BenchmarkTools, Plots

set = config_ntuple(;
    u0=-5,
    u1=5,
    nu=80,
    v0=-5,
    v1=5,
    nv=28,
    w0=-5,
    w1=5,
    nw=28,
    t1=3.4,
    nt=35,
    Kn=1,
    alpha=1.0,
    omega=0.5,
    nh=8,
)

tspan = (0.0, set.t1)
tsteps = linspace(tspan[1], tspan[2], set.nt)
γ = 5 / 3

vs = VSpace1D(set.u0, set.u1, set.nu)
vs2 = VSpace2D(set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)
vs3 = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)

f0 = @. 0.5 *
   (1 / π)^1.5 *
   (exp(-(vs3.u - 1)^2) + 0.7 * exp(-(vs3.u + 1)^2)) *
   exp(-vs3.v^2) *
   exp(-vs3.w^2)
w0 = moments_conserve(f0, vs3.u, vs3.v, vs3.w, vs3.weights)
prim0 = conserve_prim(w0, γ)
M0 = maxwellian(vs3.u, vs3.v, vs3.w, prim0)
mu_ref = ref_vhs_vis(set.Kn, set.alpha, set.omega)
τ0 = mu_ref * 2.0 * prim0[end]^(0.5) / prim0[1]

# Boltzmann
prob = ODEProblem(boltzmann_ode!, f0, tspan, fsm_kernel(vs3, mu_ref))
data_boltz = solve(prob, Tsit5(); saveat=tsteps) |> Array

# BGK
prob1 = ODEProblem(bgk_ode!, f0, tspan, (M0, τ0))
data_bgk = solve(prob1, Tsit5(); saveat=tsteps) |> Array

data_boltz_1D = zeros(axes(data_boltz, 1), axes(data_boltz, 4))
data_bgk_1D = zeros(axes(data_bgk, 1), axes(data_bgk, 4))
for j in axes(data_boltz_1D, 2)
    data_boltz_1D[:, j] .= reduce_distribution(data_boltz[:, :, :, j], vs2.weights)
    data_bgk_1D[:, j] .= reduce_distribution(data_bgk[:, :, :, j], vs2.weights)
end
f0_1D = reduce_distribution(f0, vs2.weights)
M0_1D = reduce_distribution(M0, vs2.weights)

X = Array{Float64}(undef, vs.nu, 1)
for i in axes(X, 2)
    X[:, i] .= f0_1D
end
Y = Array{Float64}(undef, vs.nu, 1, set.nt)
for i in axes(Y, 2)
    Y[:, i, :] .= data_boltz_1D
end
M = Array{Float64}(undef, vs.nu, size(X, 2))
for i in axes(M, 2)
    M[:, i] .= M0_1D
end
τ = Array{Float64}(undef, 1, size(X, 2))
for i in axes(τ, 2)
    τ[1, i] = τ0
end

nn = FnChain(FnDense(set.nu, set.nu * set.nh, tanh), FnDense(set.nu * set.nh, set.nu))
p0 = init_params(nn)

function dfdt(df, f, p, t)
    return df .= (M .- f) ./ τ .+ nn(M .- f, p) # physics
    #df .= nn(f, p) # direct
end
prob_ube = ODEProblem(dfdt, X, tspan, p0)

function loss(p)
    sol_ube = solve(prob_ube, Midpoint(); u0=X, p=p, saveat=tsteps)
    loss = sum(abs2, Array(sol_ube) .- Y)

    return loss
end

cb = function (p, l)
    display(l)
    return false
end

res = sci_train(loss, p0, Adam(); cb=cb, iters=10)
res = sci_train(loss, res.u, LBFGS(); cb=cb, iters=20, ad=AutoZygote())
res = sci_train(loss, res.u, AdamW(1e-5); cb=cb, iters=100, ad=AutoZygote())

# id
data_nn = solve(prob_ube, Tsit5(); u0=X, p=res.u, saveat=tsteps) |> Array

idx = 21
plot(
    vs.u,
    data_boltz_1D[:, 1];
    lw=1.5,
    label="initial",
    color=:gray32,
    xlabel="u",
    ylabel="particle distribution",
)
plot!(vs.u, data_boltz_1D[:, idx]; lw=1.5, label="Boltzmann", color=1)
plot!(vs.u, data_bgk_1D[:, idx]; lw=1.5, line=:dash, label="BGK", color=2)
plot!(vs.u, M; lw=1.5, label="Maxwellian", color=10)
scatter!(vs.u, data_nn[:, 1, idx]; lw=1.5, label="neural", color=3, alpha=0.7)

# ood
prob_ube1 = ODEProblem(dfdt, X, (0, 5), res.u)
data_nn = solve(prob_ube1, Tsit5(); saveat=linspace(0, 5, 51)) |> Array

plot(vs.u, M; lw=1.5, label="Maxwellian", color=10)
scatter!(vs.u, data_nn[:, 1, end]; lw=1.5, label="neural", color=3, alpha=0.7)

# output
using KitBase.JLD2
cd(@__DIR__)
u = res.u
#@save "relax_direct.jld2" u
@save "relax_phys.jld2" u
