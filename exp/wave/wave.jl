"""
Quantifying the Knuden number of the reference state in the wave propagation problem

"""

using OrdinaryDiffEq, SciMLSensitivity, Solaris
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
    mu = p[1]
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
        return KB.maxwellian(u, prim[:, i])
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

using Plots
solw0 = zeros(ps.nx, 3)
for i in axes(solw0, 1)
    _w = KA.moments_conserve(sol0[:, i, end], u, weights)
    solw0[i, :] .= KA.conserve_prim(_w, 3)
end
plot(ps.x, solw0[:, 1]; label="ρ")
plot!(ps.x, prim0[1, :])

function loss(p)
    prob = ODEProblem(rhs!, f0, tspan, p)
    sol = solve(prob, Euler(); saveat=tran, dt=dt) |> Array
    l = sum(abs2, sol .- sol0)

    return l
end

loss(p0)

cb = function (p, l)
    println("loss: $(l)")
    return false
end

res = sci_train(loss, [10.0], Adam(0.05); cb=cb, iters=200, ad=AutoZygote())
res = sci_train(loss, res.u, LBFGS(); cb=cb, iters=50, ad=AutoZygote())

@show res.u

"""~[0.00985713179259482]"""
