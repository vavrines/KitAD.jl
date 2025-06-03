using OrdinaryDiffEq, KitBase, Solaris
using CairoMakie, NipponColors
using KitBase.JLD2

D = dict_color()

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
    alpha=1.0,
    omega=0.5,
)

tspan = (0, set.t1)
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

prob = ODEProblem(boltzmann_ode!, f0, tspan, fsm_kernel(vs3, mu_ref))
data_boltz = solve(prob, Tsit5(); saveat=tsteps) |> Array

GC.gc()
@time solve(prob, Tsit5(); saveat=tsteps)
# 2.318710 seconds (38.58 k allocations: 6.348 GiB, 13.46% gc time)

prob1 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ0])
data_bgk = solve(prob1, Tsit5(); saveat=tsteps) |> Array

data_boltz_1D = zeros(axes(data_boltz, 1), axes(data_boltz, 4))
data_bgk_1D = zeros(axes(data_bgk, 1), axes(data_bgk, 4))
for j in axes(data_boltz_1D, 2)
    data_boltz_1D[:, j] .= reduce_distribution(data_boltz[:, :, :, j], vs2.weights)
    data_bgk_1D[:, j] .= reduce_distribution(data_bgk[:, :, :, j], vs2.weights)
end
h0_1D = reduce_distribution(f0, vs2.weights)
H0_1D = reduce_distribution(M0, vs2.weights)
nn = FnChain(FnDense(vs.nu, vs.nu * 8, tanh), FnDense(vs.nu * 8, vs.nu))
p0 = init_params(nn)

cd(@__DIR__)
@load "relax_direct.jld2" u
#@load "relax_phys.jld2" u

function rhs(f, p, t)
    #    return (H0_1D .- f) ./ τ0 .+ nn(H0_1D .- f, p)
    return nn(f, p)
end

ube = ODEProblem(rhs, h0_1D, tspan, p0)
data_nn = solve(ube, Tsit5(); u0=h0_1D, p=u, saveat=tsteps) |> Array

GC.gc()
@time solve(ube, Tsit5(); u0=h0_1D, p=u, saveat=tsteps)
# 0.003911 seconds (8.16 k allocations: 47.943 MiB)

function rhs11(f, p, t)
    return (H0_1D .- f) ./ τ0
end
ube1 = ODEProblem(rhs11, h0_1D, tspan, p0)
GC.gc()
@time solve(ube1, Tsit5(); u0=h0_1D, p=u, saveat=tsteps)
# 0.000896 seconds (4.57 k allocations: 1.373 MiB)

begin
    idx = 2
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="u", ylabel="Distribution")
    CairoMakie.scatter!(ax, vs.u, data_nn[:, idx]; color=D["aonibi"], label="optimized")
    lines!(ax, vs.u, data_boltz_1D[:, idx]; color=D["asagi"], label="Boltzmann")
    lines!(ax, vs.u, data_bgk_1D[:, idx]; color=D["tohoh"], label="BGK")
    axislegend(; position=:rt)
    f
end
save("relax_t1.pdf", f)

begin
    idx = 4
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="u", ylabel="Distribution")
    scatter!(ax, vs.u, data_nn[:, idx]; color=D["aonibi"], label="optimized")
    lines!(ax, vs.u, data_boltz_1D[:, idx]; color=D["asagi"], label="Boltzmann")
    lines!(ax, vs.u, data_bgk_1D[:, idx]; color=D["tohoh"], label="BGK")
    axislegend(; position=:rt)
    f
end
save("relax_t2.pdf", f)

begin
    idx = 6
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="u", ylabel="Distribution")
    scatter!(ax, vs.u, data_nn[:, idx]; color=D["aonibi"], label="optimized")
    lines!(ax, vs.u, data_boltz_1D[:, idx]; color=D["asagi"], label="Boltzmann")
    lines!(ax, vs.u, data_bgk_1D[:, idx]; color=D["tohoh"], label="BGK")
    axislegend(; position=:rt)
    f
end
save("relax_t3.pdf", f)

begin
    idx = 11
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="u", ylabel="Distribution")
    CairoMakie.scatter!(ax, vs.u, data_nn[:, idx]; color=D["aonibi"], label="optimized")
    lines!(ax, vs.u, data_boltz_1D[:, idx]; color=D["asagi"], label="Boltzmann")
    lines!(ax, vs.u, data_bgk_1D[:, idx]; color=D["tohoh"], label="BGK")
    axislegend(; position=:rt)
    f
end
save("relax_t4.pdf", f)

begin
    idx = 21
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="u", ylabel="Distribution")
    scatter!(ax, vs.u, data_nn[:, idx]; color=D["aonibi"], label="optimized")
    lines!(ax, vs.u, data_boltz_1D[:, idx]; color=D["asagi"], label="Boltzmann")
    lines!(ax, vs.u, data_bgk_1D[:, idx]; color=D["tohoh"], label="BGK")
    axislegend(; position=:rt)
    f
end
save("relax_t5.pdf", f)

begin
    idx = 31
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="u", ylabel="Distribution")
    scatter!(ax, vs.u, data_nn[:, idx]; color=D["aonibi"], label="optimized")
    lines!(ax, vs.u, data_boltz_1D[:, idx]; color=D["asagi"], label="Boltzmann")
    lines!(ax, vs.u, data_bgk_1D[:, idx]; color=D["tohoh"], label="BGK")
    axislegend(; position=:rt)
    f
end
save("relax_t6.pdf", f)

begin
    idx = 41
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="u", ylabel="Distribution")
    scatter!(ax, vs.u, data_nn[:, idx]; color=D["aonibi"], label="optimized")
    lines!(ax, vs.u, data_boltz_1D[:, idx]; color=D["asagi"], label="Boltzmann")
    lines!(ax, vs.u, data_bgk_1D[:, idx]; color=D["tohoh"], label="BGK")
    axislegend(; position=:rt)
    f
end
save("relax_o1.pdf", f)

begin
    idx = 51
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="u", ylabel="Distribution")
    scatter!(ax, vs.u, data_nn[:, idx]; color=D["aonibi"], label="optimized")
    lines!(ax, vs.u, data_boltz_1D[:, idx]; color=D["asagi"], label="Boltzmann")
    lines!(ax, vs.u, data_bgk_1D[:, idx]; color=D["tohoh"], label="BGK")
    axislegend(; position=:rt)
    f
end
save("relax_o2.pdf", f)

begin
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="u", ylabel="t", title="", aspect=1)
    co = contourf!(vs.u, tsteps, data_nn; colormap=:PiYG_8, levels=20)
    Colorbar(fig[1, 2], co)
    fig
end
save("relax_pdf.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="u", ylabel="t", title="", aspect=1)
    co = contourf!(vs.u, tsteps, data_nn .- data_bgk_1D; colormap=:PiYG_8, levels=15)
    Colorbar(fig[1, 2], co)
    fig
end
save("relax_df.pdf", fig)

r = rhs(data_nn, u, 0.0)
rhs0(f, p, t) = (H0_1D .- f) ./ τ0
r0 = rhs0(data_bgk_1D, u, 0.0)

begin
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="u", ylabel="t", title="", aspect=1)
    co = contourf!(vs.u, tsteps, r; colormap=:PiYG_8, levels=20)
    Colorbar(fig[1, 2], co)
    fig
end
save("relax_q.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="u", ylabel="t", title="", aspect=1)
    co = contourf!(vs.u, tsteps, r .- r0; colormap=:PiYG_8, levels=20)
    Colorbar(fig[1, 2], co)
    fig
end
save("relax_dq.pdf", fig)
