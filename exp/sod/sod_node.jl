using OrdinaryDiffEq, SciMLSensitivity, Solaris
using Solaris.Lux
import KitAD as KA
import KitBase as KB

const γ = 5 / 3

tspan = (0.0, 0.2)
dt = 0.001
tran = tspan[1]:dt*10:tspan[end]
ps = KB.PSpace1D(0, 1, 100)
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

model = Chain(Dense(300, 300, tanh), Dense(300, 300))
ps, st = SR.setup(model)
ps = ComponentArray(ps)
model = Lux.StatefulLuxLayer{true}(model, ps, st)

function rhs!(dw, w, p, t)
    dv = model(w[1:300], p)
    dw .= reshape(dv, 3, 100)

    return nothing
end

prob0 = ODEProblem(rhs!, w0, tspan, ps)
sol0 = solve(prob0, Tsit5(); saveat=tspan[2]) |> Array

function loss(p)
    prob = ODEProblem(rhs!, w0, tspan, p)
    sol = solve(prob, Euler(); saveat=tspan[end], dt=dt)
    l = sum(abs2, sol.u[end] .- sole_cons)

    return l
end

loss(ps)

cb = function (p, l)
    println("loss: $(l)")
    return false
end

res = sci_train(loss, ps, Adam(0.05); cb=cb, iters=100, ad=AutoZygote())
res = sci_train(loss, res.u, AdamW(0.01); cb=cb, iters=100, ad=AutoZygote())

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
    plot(ps.x[1:ps.nx], solprim1[:, idx]; label="neural")
    plot!(ps.x[1:ps.nx], sole_prim[idx, :]; label="exact")
end
