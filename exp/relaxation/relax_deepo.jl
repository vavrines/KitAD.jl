using KitBase, OrdinaryDiffEq, SciMLSensitivity, Solaris, NeuralOperators
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
    t1=8,
    nt=81,
    Kn=1,
)

tspan = (0, set.t1)
tsteps = linspace(tspan[1], tspan[2], set.nt)
γ = 5 / 3

vs = VSpace1D(set.u0, set.u1, set.nu)
vs2 = VSpace2D(set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)
vs3 = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)

f0 =
    0.5 * (1 / π)^1.5 .* (exp.(-(vs3.u .- 1) .^ 2) .+ 0.7 .* exp.(-(vs3.u .+ 1) .^ 2)) .*
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
data_boltz = solve(prob, Tsit5(); saveat=tsteps) |> Array

prob1 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ0])
data_bgk = solve(prob1, Tsit5(); saveat=tsteps) |> Array

data_boltz_1D = zeros(axes(data_boltz, 1), axes(data_boltz, 4))
data_bgk_1D = zeros(axes(data_bgk, 1), axes(data_bgk, 4))
for j in axes(data_boltz_1D, 2)
    data_boltz_1D[:, j] .= reduce_distribution(data_boltz[:, :, :, j], vs2.weights)
    data_bgk_1D[:, j] .= reduce_distribution(data_bgk[:, :, :, j], vs2.weights)
end
h0_1D, b0_1D = reduce_distribution(f0, vs3.v, vs3.w, vs2.weights)
H0_1D, B0_1D = reduce_distribution(M0, vs3.v, vs3.w, vs2.weights)

#nn1 = FnChain(FnDense(vs.nu, vs.nu, tanh), FnDense(vs.nu, vs.nu, tanh), FnDense(vs.nu, vs.nu))
#nn2 = FnChain(FnDense(1, vs.nu, tanh), FnDense(vs.nu, vs.nu, tanh), FnDense(vs.nu, vs.nu))
nn1 = FnChain(FnDense(vs.nu, vs.nu, tanh), FnDense(vs.nu, vs.nu))
nn2 = FnChain(FnDense(1, vs.nu, tanh), FnDense(vs.nu, vs.nu))
nn = FnChain(nn1, nn2)
p0 = init_params(nn)

ys = reshape(vs.u, 1, :)

function infer_deeponet(model, u, y, p)
    bnet = model.layers[1]
    tnet = model.layers[2]
    np1 = length(init_params(bnet))

    bs = bnet(u, p[1:np1])
    ts = tnet(y, p[np1+1:end])
    pred = permutedims(bs) * ts

    return pred
end

M = H0_1D
function rhs(f, p, t)
    return (M .- f) ./ τ0 .+ infer_deeponet(nn, M .- f, ys, p)[:]
end

exact = data_boltz_1D
function loss(p)
    ube = ODEProblem(rhs, h0_1D, tspan, p)
    pred = solve(ube, Midpoint(); saveat=tsteps) |> Array
    loss = sum(abs2, exact .- pred)

    return loss
end

@time loss(p0)

res = sci_train(loss, p0, Adam(0.05); cb=default_callback, iters=10, ad=AutoZygote())
res = sci_train(loss, res.u, LBFGS(); cb=default_callback, iters=10, ad=AutoZygote())
res = sci_train(loss, res.u, AdamW(0.001); cb=default_callback, iters=20, ad=AutoZygote())

# ~0.0008

using KitBase.JLD2
u = res.u
@save "deepo.jld2" u
