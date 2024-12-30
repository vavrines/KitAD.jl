"""
Quantifying the Knuden number of the reference state in the homegenous relaxation problem of non-equilibrium Argon gas

"""

using OrdinaryDiffEq, SciMLSensitivity, Solaris
using Optimization: AutoZygote
using Optimisers: Adam
using Optim: LBFGS
import KitBase as KB

cd(@__DIR__)

tspan = (0.0, 3.0)
dt = 0.1
tran = tspan[1]:dt:tspan[end]
vs = KB.VSpace1D(-6, 6, 40)

f0 = 0.5 * (1 / π)^1.5 .* (exp.(-(vs.u .- 1) .^ 2) .+ exp.(-(vs.u .+ 1) .^ 2))
w0 = KB.moments_conserve(f0, vs.u, vs.weights)
prim0 = KB.conserve_prim(w0, 3)
M0 = KB.maxwellian(vs.u, prim0)
Kn0 = 1.0
μ0 = KB.ref_vhs_vis(Kn0, 1.0, 0.5)
τ0 = μ0 * 2.0 * prim0[end]^(0.5) / prim0[1]
p0 = [τ0]

function bgk!(df, f, p, t)
    w = KB.moments_conserve(f, vs.u, vs.weights)
    prim = KB.conserve_prim(w, 3)
    M = KB.maxwellian(vs.u, prim)
    tau = p[1]
    return df .= (M .- f) ./ tau
end

prob0 = ODEProblem(bgk!, f0, tspan, p0)
sol0 = solve(prob0, Tsit5(); saveat=tran) |> Array

function loss(p)
    prob = ODEProblem(bgk!, f0, tspan, p)
    sol = solve(prob, Tsit5(); saveat=tran) |> Array
    l = sum(abs2, sol .- sol0)

    return l
end

cb = function (p, l)
    println("loss: $(l)")
    return false
end

res = sci_train(loss, [10.0], Adam(); cb=cb, iters=1000, ad=AutoZygote())
res = sci_train(loss, res.u, LBFGS(); cb=cb, iters=1000, ad=AutoZygote())

@show res.u
