using OrdinaryDiffEq, CairoMakie, NipponColors
import KitAD as KA
import KitBase as KB

cd(@__DIR__)
D = dict_color()

begin
    tspan = (0.0, 0.25)
    dt = 0.001
    tran = tspan[1]:dt*10:tspan[end]
    ps = KB.PSpace1D(0, 1, 100, 0)
    dx = ps.dx[1]
    vs = KB.VSpace1D(-5, 5, 28)
    u = vs.u
    weights = vs.weights
end

begin
    prim0 = zeros(3, axes(ps.x, 1))
    f0 = zeros(vs.nu, axes(ps.x, 1))
    for i in axes(f0, 2)
        ρ = 1 + 0.1 * sin(2π * ps.x[i])
        λ = ρ
        prim0[:, i] .= [ρ, 1.0, λ]
        f0[:, i] .= KB.maxwellian(vs.u, prim0[:, i])
    end
end

begin
    Kn0 = 1e-2
    μ0 = KB.ref_vhs_vis(Kn0, 1.0, 0.5)
    τ0 = μ0 * 2.0 * prim0[end]^(0.5) / prim0[1]
    p0 = [μ0]
end

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

begin
    p0 = [0.01]
    prob0 = ODEProblem(rhs!, f0, tspan, p0)
    sol0 = solve(prob0, Tsit5(); saveat=tran) |> Array

    prob1 = ODEProblem(rhs!, f0, tspan, [10.0])
    sol1 = solve(prob1, Tsit5(); saveat=tran) |> Array

    prob2 = ODEProblem(rhs!, f0, tspan, [0.00985713179259482])
    sol2 = solve(prob2, Tsit5(); saveat=tran) |> Array
end

begin
    solw0 = zeros(ps.nx, 3)
    solw1 = deepcopy(solw0)
    solw2 = deepcopy(solw0)
    for i in axes(solw0, 1)
        _w = KA.moments_conserve(sol0[:, i, end], u, weights)
        solw0[i, :] .= KA.conserve_prim(_w, 3)
        _w = KA.moments_conserve(sol1[:, i, end], u, weights)
        solw1[i, :] .= KA.conserve_prim(_w, 3)
        _w = KA.moments_conserve(sol2[:, i, end], u, weights)
        solw2[i, :] .= KA.conserve_prim(_w, 3)

        solw0[i, 3] = 1 / solw0[i, 3]
        solw1[i, 3] = 1 / solw1[i, 3]
        solw2[i, 3] = 1 / solw2[i, 3]
    end
end

begin
    idx = 1
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="Density")
    lines!(ax, ps.x[1:ps.nx], solw1[:, idx]; color=D["asagi"], label="initial")
    lines!(ax, ps.x[1:ps.nx], solw2[:, idx]; color=D["tohoh"], label="optimized")
    lines!(
        ax,
        ps.x[1:ps.nx],
        solw0[:, idx];
        color=D["ro"],
        linestyle=:dash,
        label="reference",
    )
    axislegend(; position=:rt)
    f
end
#save("wave_density.pdf", f)

begin
    idx = 2
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="Velocity")
    lines!(ax, ps.x[1:ps.nx], solw1[:, idx]; color=D["asagi"], label="initial")
    lines!(ax, ps.x[1:ps.nx], solw2[:, idx]; color=D["tohoh"], label="optimized")
    lines!(
        ax,
        ps.x[1:ps.nx],
        solw0[:, idx];
        color=D["ro"],
        linestyle=:dash,
        label="reference",
    )
    axislegend(; position=:lt)
    f
end
#save("wave_velocity.pdf", f)

begin
    idx = 3
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="Temperature")
    lines!(ax, ps.x[1:ps.nx], solw1[:, idx]; color=D["asagi"], label="initial")
    lines!(ax, ps.x[1:ps.nx], solw2[:, idx]; color=D["tohoh"], label="optimized")
    lines!(
        ax,
        ps.x[1:ps.nx],
        solw0[:, idx];
        color=D["ro"],
        linestyle=:dash,
        label="reference",
    )
    axislegend(; position=:rb)
    f
end
#save("wave_temperature.pdf", f)
