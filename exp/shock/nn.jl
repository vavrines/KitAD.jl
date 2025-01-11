using KitBase, OrdinaryDiffEq, SciMLSensitivity, Solaris
using KitBase.ProgressMeter: @showprogress

set = Setup(; case="shock", space="1d2f1v", collision="bgk")
ps = PSpace1D(-25, 25, 50)
vs = VSpace1D(-8, 8, 36)
gas = Gas(; Kn=1.0, Ma=3.0, K=2.0)
ib = IB2F(KB.config_ib(set, ps, vs, gas)...)
ks = SolverSet(set, ps, vs, gas, ib)

using KitBase.JLD2
cd(@__DIR__)
@load "shakhov.jld2" ctr
solf0 = zeros(Float32, vs.nu * 2, ps.nx)
for i in 1:ps.nx
    solf0[1:vs.nu, i] .= ctr[i].h
    solf0[vs.nu+1:end, i] .= ctr[i].b
end

begin
    tspan = (0.0, 50.0)
    dt = 0.1
    dx = ps.dx[1]
    nx = ps.nx
    u = vs.u
    nu = vs.nu
    weights = vs.weights
    inK = gas.K
    mu = gas.μᵣ
end

begin
    prim0 = zeros(Float32, 3, axes(ps.x, 1))
    f0 = zeros(Float32, vs.nu * 2, ps.nx)
    for i in axes(f0, 2)
        prim0[:, i] .= ib.bc(ps.x[i], ib.p)
        f0[1:vs.nu, i] .= KB.maxwellian(vs.u, prim0[:, i])
        f0[vs.nu+1:end, i] .= KB.energy_maxwellian(f0[1:vs.nu, i], prim0[:, i], gas.K)
    end
end

function rhs!(df, f, p, t)
    nu2 = size(f, 1)
    nu = nu2 ÷ 2
    nx = size(f, 2)

    h = f[1:nu, :]
    b = f[nu+1:nu*2, :]

    fluxh = zeros(nu, nx + 1)
    fluxb = zeros(nu, nx + 1)
    for j in 2:nx
        for i in 1:nu÷2
            fluxh[i, j] = u[i] * h[i, j]
            fluxb[i, j] = u[i] * b[i, j]
        end
        for i in nu÷2+1:nu
            fluxh[i, j] = u[i] * h[i, j-1]
            fluxb[i, j] = u[i] * b[i, j-1]
        end
    end

    #flux = vcat(fluxh, fluxb)
    flux = zeros(nu2, nx + 1)
    for j in 2:nx
        for i in 1:nu
            flux[i, j] = fluxh[i, j]
            flux[nu+i, j] = fluxb[i, j]
        end
    end

    w = reduce(hcat, map(1:nx) do i
        return KA.moments_conserve(h[:, i], b[:, i], u, weights)
    end)
    prim = reduce(hcat, map(1:nx) do i
        return KA.conserve_prim(w[:, i], 5 / 3)
    end)
    H = reduce(hcat, map(1:nx) do i
        return KA.maxwellian(u, prim[:, i])
    end)
    B = reduce(hcat, map(1:nx) do i
        return KA.energy_maxwellian(H[:, i], prim[end, i], inK)
    end)

    #M = vcat(H, B)
    M = zeros(nu2, nx)
    for j in axes(f, 2)
        for i in 1:nu
            M[i, j] = H[i, j]
            M[nu+i, j] = B[i, j]
        end
    end

    for j in 2:nx-1
        tau = mu * 2.0 * prim[end, j]^0.5 / prim[1, j]
        for i in axes(f, 1)
            df[i, j] = (flux[i, j] - flux[i, j+1]) / dx + (M[i, j] - f[i, j]) / tau
            df[i, 1] = 0.0
            df[i, nx] = 0.0
        end

        dq = nn(M .- f, p)
        for i in axes(f, 1)
            df[i, j] += dq[i, j] * 0.01
        end
    end

    return nothing
end

nn = FnChain(
    FnDense(vs.nu * 2, vs.nu * 2, tanh; bias=false),
    FnDense(vs.nu * 2, vs.nu * 2; bias=false),
)
p0 = init_params(nn)

prob0 = ODEProblem(rhs!, f0, tspan, p0)
sol0 = solve(prob0, Euler(); dt=dt, saveat=tspan[2])[2]

function loss(p)
    prob = ODEProblem(rhs!, f0, tspan, p)
    sol = solve(prob, Euler(); saveat=tspan[2], dt=dt)[2]

    #solw = reduce(hcat, map(1:nx) do i
    #    return KA.moments_conserve(sol[1:nu, i, end], sol[nu+1:end, i, end], u, weights)
    #end)

    l = sum(abs2, sol .- solf0)

    return l
end

loss(p0)
GC.gc()

res = sci_train(loss, p0, Adam(0.05); cb=default_callback, iters=50, ad=AutoZygote())
res = sci_train(loss, res.u, AdamW(0.005); cb=default_callback, iters=50, ad=AutoZygote())
