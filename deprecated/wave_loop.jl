"""
Quantifying the Knuden number of the reference state in the wave propagation problem
Simple for-loop is used with no SciMLSensitivity involved

"""

using OrdinaryDiffEq, Zygote, Solaris
using Optimization: AutoZygote
using Optimisers: Adam
using Optim: LBFGS
import KitBase as KB
import KitAD as KA

tspan = (0.0, 0.01)
dt = 0.0005
tran = tspan[1]:dt:tspan[end]
ps = KB.PSpace1D(0, 1, 50, 0)
dx = ps.dx[1]
vs = KB.VSpace1D(-5, 5, 28)
u = vs.u
weights = vs.weights

Kn0 = 1e-2
μ0 = KB.ref_vhs_vis(Kn0, 1.0, 0.5)
τ0 = μ0 * 2.0 * prim0[end]^(0.5) / prim0[1]
p0 = [μ0]

prim0 = zeros(3, axes(ps.x, 1))
f0 = zeros(vs.nu, axes(ps.x, 1))
for i in axes(f0, 2)
    _ρ = 1 + 0.05 * sin(2π * ps.x[i])
    _λ = _ρ
    prim0[:, i] .= [_ρ, 1.0, _λ]
    f0[:, i] .= KB.maxwellian(vs.u, prim0[:, i])
end

function rhs(f, p, t)
    mu = p[1]
    nu = size(f, 1)
    nx = size(f, 2)

    flux = reduce(hcat, map(2:nx) do i
        KA.flux_kfvs.(f[:, i-1], f[:, i], u)
    end)
    flux = hcat(KA.flux_kfvs.(f[:, nx], f[:, 1], u), flux)
    flux = hcat(flux, KA.flux_kfvs.(f[:, nx], f[:, 1], u))

    w = reduce(hcat, map(1:nx) do i
        KA.moments_conserve(f[:, i], u, weights)
    end)
    prim = reduce(hcat, map(1:nx) do i
        KA.conserve_prim(w[:, i], 3)
    end)
    M = reduce(hcat, map(1:nx) do i
        KB.maxwellian(u, prim[:, i])
    end)
    tau = reduce(hcat, map(1:nx) do i
        mu * 2.0 * prim[end, i]^0.5 / prim[1, i]
    end)

    df = @. (flux[:, 1:nx] - flux[:, 2:nx+1]) / dx + (M - f) / tau

    return df
end

prob0 = ODEProblem(rhs, f0, tspan, p0)
sol0 = solve(prob0, Euler(), saveat = tran, dt = dt) |> Array
solf = sol0[:, :, end]

function loss(p)
    f = deepcopy(f0)
    for iter = 1:20
        fn = deepcopy(f)
        df = rhs(f, p, 0.0)
        f = @. fn + df * dt
    end
    l = sum(abs2, f .- solf)

    return l
end

#Zygote.pullback(loss, p0)[2](1)

cb = function (p, l)
    println("loss: $(loss(p))")
    return false
end

res = sci_train(loss, [1.0], Adam(); cb = cb, iters = 1000, ad = AutoZygote())

@show res.u
