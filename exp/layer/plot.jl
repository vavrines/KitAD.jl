"""
Machine learning assisted modeling for hydrodynamic closure

"""

using OrdinaryDiffEq, KitBase, Solaris, CairoMakie, NipponColors
using KitBase.JLD2
using Solaris.Flux: sigmoid

D = dict_color()

function flux_basis(wL, wR)
    primL = conserve_prim(wL, γ)
    primR = conserve_prim(wR, γ)
    Mu1, Mv1, Mw1, MuL1, MuR1 = KB.gauss_moments(primL, 1)
    Mu2, Mv2, Mw2, MuL2, MuR2 = KB.gauss_moments(primR, 1)
    MuvL = moments_conserve(MuL1, Mv1, Mw1, 1, 0, 0)
    MuvR = moments_conserve(MuR2, Mv2, Mw2, 1, 0, 0)
    fw = primL[1] * MuvL + primR[1] * MuvR

    return fw
end

cd(@__DIR__)
@load "layer_ktsol.jld2" resArr

nn = FnChain(FnDense(8, 32, tanh), FnDense(32, 4, tanh))
p0 = init_params(nn) #.|> Float64

ps = PSpace1D(-0.1, 0.1, 80, 0)
gas = Gas(; Kn=5e-3, K=1.0)

w0 = zeros(Float32, 4, ps.nx)
prim0 = zeros(Float32, 4, ps.nx)
for i in 1:ps.nx
    if ps.x[i] <= 0
        prim0[:, i] .= [1.0, 0.0, 1.0, 1.0]
    else
        prim0[:, i] .= [1.0, 0.0, -1.0, 2.0]
    end

    w0[:, i] .= prim_conserve(prim0[:, i], gas.γ)
end

begin
    τ0 = vhs_collision_time(prim0[:, 1], gas.μᵣ, gas.ω)
    tmax = 10τ0
    tspan = (0.0, tmax)
    dt = tmax / 20 / 10
    tran = tspan[1]:dt*10:tspan[end]
    dx = ps.dx[1]
    K = gas.K
    γ = gas.γ
    μᵣ = gas.μᵣ
    ω = gas.ω
end

function rhs0!(dw, w, p, t)
    nx = size(w, 2)

    flux = zeros(Float32, 4, nx + 1)
    for j in 2:nx
        flux[:, j] .= flux_basis(w[:, j-1], w[:, j])
    end

    for j in 2:nx-1
        for i in 1:4
            dw[i, j] = (flux[i, j] - flux[i, j+1]) / dx
        end
    end

    return nothing
end
function rhs!(dw, w, p, t)
    nx = size(w, 2)

    flux = zeros(Float32, 4, nx + 1)
    for j in 2:nx
        flux[:, j] .= flux_basis(w[:, j-1], w[:, j])
        flux[:, j] .+= nn(vcat(w[:, j-1], w[:, j]), p) .* 0.1
    end

    for j in 2:nx-1
        for i in 1:4
            dw[i, j] = (flux[i, j] - flux[i, j+1]) / dx
            dw[i, 1] = 0.0
            dw[i, nx] = 0.0
        end
    end

    return nothing
end

@load "layer_param.jld2" u

prob = ODEProblem(rhs!, w0, tspan, u)
sol = solve(prob, Euler(); saveat=tran, dt=dt) |> Array
#sol = solve(prob, Tsit5(); saveat=tran) |> Array

prob0 = ODEProblem(rhs0!, w0, tspan, u)
sol0 = solve(prob0, Euler(); saveat=tran, dt=dt) |> Array
#sol0 = solve(prob0, Tsit5(); saveat=tran) |> Array

# benchmark
@time KB.flux_gks!(
    zeros(4),
    w0[:, 1],
    w0[:, 2],
    gas.K,
    gas.γ,
    gas.μᵣ,
    gas.ω,
    1.0,
    0.1,
    0.1,
    0.1,
    zero(w0[:, 1]),
    zero(w0[:, 1]),
)
@time nn(vcat(w0[:, 1], w0[:, 2]), p0)


solp = zero(sol)
solp_ns = zero(sol)
solp_kt = zero(sol)
for i in 1:ps.nx, j in 1:length(tran)
    solp[:, i, j] .= conserve_prim(sol[:, i, j], γ)
    solp_ns[:, i, j] .= conserve_prim(sol0[:, i, j], γ)
    solp_kt[:, i, j] .= conserve_prim(resArr[:, i, j], γ)

    solp[end, i, j] = 1 / solp[end, i, j]
    solp_ns[end, i, j] = 1 / solp_ns[end, i, j]
    solp_kt[end, i, j] = 1 / solp_kt[end, i, j]
end

begin
    idx, itx = 1, 3
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="Density")
    scatter!(ax, ps.x, solp[idx, :, itx]; color=D["aonibi"], label="current")
    lines!(ax, ps.x, solp_kt[idx, :, itx]; color=D["asagi"], label="kinetic")
    lines!(ax, ps.x, solp_ns[idx, :, itx]; color=D["tohoh"], label="continuum")
    axislegend(; position=:lt)
    f
    save("layer1tau_density.pdf", f)

    idx, itx = 2, 3
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="U")
    scatter!(ax, ps.x, solp[idx, :, itx]; color=D["aonibi"], label="current")
    lines!(ax, ps.x, solp_kt[idx, :, itx]; color=D["asagi"], label="kinetic")
    lines!(ax, ps.x, solp_ns[idx, :, itx]; color=D["tohoh"], label="continuum")
    axislegend(; position=:lt)
    f
    save("layer1tau_u.pdf", f)

    idx, itx = 3, 3
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="V")
    scatter!(ax, ps.x, solp[idx, :, itx]; color=D["aonibi"], label="current")
    lines!(ax, ps.x, solp_kt[idx, :, itx]; color=D["asagi"], label="kinetic")
    lines!(ax, ps.x, solp_ns[idx, :, itx]; color=D["tohoh"], label="continuum")
    axislegend(; position=:lt)
    f
    save("layer1tau_v.pdf", f)

    idx, itx = 4, 3
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="Temperature")
    scatter!(ax, ps.x, solp[idx, :, itx]; color=D["aonibi"], label="current")
    lines!(ax, ps.x, solp_kt[idx, :, itx]; color=D["asagi"], label="kinetic")
    lines!(ax, ps.x, solp_ns[idx, :, itx]; color=D["tohoh"], label="continuum")
    axislegend(; position=:lt)
    f
    save("layer1tau_temperature.pdf", f)
end

begin
    idx, itx = 1, 11
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="Density")
    scatter!(ax, ps.x, solp[idx, :, itx]; color=D["aonibi"], label="current")
    lines!(ax, ps.x, solp_kt[idx, :, itx]; color=D["asagi"], label="kinetic")
    lines!(ax, ps.x, solp_ns[idx, :, itx]; color=D["tohoh"], label="continuum")
    axislegend(; position=:lt)
    f
    save("layer5tau_density.pdf", f)

    idx, itx = 2, 11
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="U")
    scatter!(ax, ps.x, solp[idx, :, itx]; color=D["aonibi"], label="current")
    lines!(ax, ps.x, solp_kt[idx, :, itx]; color=D["asagi"], label="kinetic")
    lines!(ax, ps.x, solp_ns[idx, :, itx]; color=D["tohoh"], label="continuum")
    axislegend(; position=:lt)
    f
    save("layer5tau_u.pdf", f)

    idx, itx = 3, 11
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="V")
    scatter!(ax, ps.x, solp[idx, :, itx]; color=D["aonibi"], label="current")
    lines!(ax, ps.x, solp_kt[idx, :, itx]; color=D["asagi"], label="kinetic")
    lines!(ax, ps.x, solp_ns[idx, :, itx]; color=D["tohoh"], label="continuum")
    axislegend(; position=:lt)
    f
    save("layer5tau_v.pdf", f)

    idx, itx = 4, 11
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="Temperature")
    scatter!(ax, ps.x, solp[idx, :, itx]; color=D["aonibi"], label="current")
    lines!(ax, ps.x, solp_kt[idx, :, itx]; color=D["asagi"], label="kinetic")
    lines!(ax, ps.x, solp_ns[idx, :, itx]; color=D["tohoh"], label="continuum")
    axislegend(; position=:lt)
    f
    save("layer5tau_temperature.pdf", f)
end

begin
    idx, itx = 1, 21
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="Density")
    scatter!(ax, ps.x, solp[idx, :, itx]; color=D["aonibi"], label="current")
    lines!(ax, ps.x, solp_kt[idx, :, itx]; color=D["asagi"], label="kinetic")
    lines!(ax, ps.x, solp_ns[idx, :, itx]; color=D["tohoh"], label="continuum")
    axislegend(; position=:lt)
    f
    save("layer10tau_density.pdf", f)

    idx, itx = 2, 21
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="U")
    scatter!(ax, ps.x, solp[idx, :, itx]; color=D["aonibi"], label="current")
    lines!(ax, ps.x, solp_kt[idx, :, itx]; color=D["asagi"], label="kinetic")
    lines!(ax, ps.x, solp_ns[idx, :, itx]; color=D["tohoh"], label="continuum")
    axislegend(; position=:lt)
    f
    save("layer10tau_u.pdf", f)

    idx, itx = 3, 21
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="V")
    scatter!(ax, ps.x, solp[idx, :, itx]; color=D["aonibi"], label="current")
    lines!(ax, ps.x, solp_kt[idx, :, itx]; color=D["asagi"], label="kinetic")
    lines!(ax, ps.x, solp_ns[idx, :, itx]; color=D["tohoh"], label="continuum")
    axislegend(; position=:lt)
    f
    save("layer10tau_v.pdf", f)

    idx, itx = 4, 21
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="Temperature")
    scatter!(ax, ps.x, solp[idx, :, itx]; color=D["aonibi"], label="current")
    lines!(ax, ps.x, solp_kt[idx, :, itx]; color=D["asagi"], label="kinetic")
    lines!(ax, ps.x, solp_ns[idx, :, itx]; color=D["tohoh"], label="continuum")
    axislegend(; position=:lt)
    f
    save("layer10tau_temperature.pdf", f)
end
