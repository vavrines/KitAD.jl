using KitBase, CairoMakie, NipponColors
using KitBase.JLD2

function extract_pdf(ks, ctr)
    ps, vs = ks.ps, ks.vs
    f = zeros(ps.nx, vs.nu)
    for i in 1:ps.nx
        f[i, :] .= ctr[i].h
    end
    return f
end

function extract_q(ks, ctr)
    ps, vs = ks.ps, ks.vs
    q = zeros(ps.nx, vs.nu)
    for i in 1:ps.nx
        H = maxwellian(vs.u, ctr[i].prim)
        tau = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
        q[i, :] .= (H .- ctr[i].h) ./ tau
    end
    return q
end

D = dict_color()

set = Setup(; case="shock", space="1d2f1v", collision="bgk", maxTime=50)
ps = PSpace1D(-25, 25, 50, 1)
vs = VSpace1D(-8, 8, 36)
gas = Gas(; Kn=1.0, Ma=3.0, K=2.0)
ib = IB2F(KB.config_ib(set, ps, vs, gas)...)
ks = SolverSet(set, ps, vs, gas, ib)

cd(@__DIR__)
@load "bgk.jld2" ctr
solb = extract_sol(ks, ctr)
solb[:, 3] .= 1 ./ solb[:, 3]
fb = extract_pdf(ks, ctr)
qb = extract_q(ks, ctr)

@load "shakhov.jld2" ctr
sols = extract_sol(ks, ctr)
sols[:, 3] .= 1 ./ sols[:, 3]
fs = extract_pdf(ks, ctr)
qs = extract_q(ks, ctr)

@load "nn.jld2" ctr
soln = extract_sol(ks, ctr)
soln[:, 3] .= 1 ./ soln[:, 3]
fn = extract_pdf(ks, ctr)
qn = extract_q(ks, ctr)

xs = ps.x[1:ps.nx]

begin
    idx = 1
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="Density")
    scatter!(ax, xs, soln[:, idx]; color=D["aonibi"], label="current")
    lines!(ax, xs, solb[:, idx]; color=D["asagi"], label="BGK")
    lines!(ax, xs, sols[:, idx]; color=D["tohoh"], label="Shakhov")
    axislegend(; position=:lt)
    f
end
save("shock_density.pdf", f)

begin
    idx = 2
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="Velocity")
    scatter!(ax, xs, soln[:, idx]; color=D["aonibi"], label="current")
    lines!(ax, xs, solb[:, idx]; color=D["asagi"], label="BGK")
    lines!(ax, xs, sols[:, idx]; color=D["tohoh"], label="Shakhov")
    axislegend(; position=:lt)
    f
end
save("shock_velocity.pdf", f)

begin
    idx = 3
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="x", ylabel="Temperature")
    scatter!(ax, xs, soln[:, idx]; color=D["aonibi"], label="current")
    lines!(ax, xs, solb[:, idx]; color=D["asagi"], label="BGK")
    lines!(ax, xs, sols[:, idx]; color=D["tohoh"], label="Shakhov")
    axislegend(; position=:lt)
    f
end
save("shock_temperature.pdf", f)

begin
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="u", title="", aspect=1)
    co = contourf!(xs, vs.u, fb; colormap=:PiYG_8, levels=20)
    Colorbar(fig[1, 2], co)
    fig
end
save("shock_pdf.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="u", title="", aspect=1)
    co = contourf!(xs, vs.u, qs; colormap=:PiYG_8, levels=20)
    Colorbar(fig[1, 2], co)
    fig
end
save("shock_q.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="u", title="", aspect=1)
    co = contourf!(xs, vs.u, fs .- fb; colormap=:PiYG_8, levels=20)
    Colorbar(fig[1, 2], co)
    fig
end
save("shock_df.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="u", title="", aspect=1)
    co = contourf!(xs, vs.u, qs .- qb; colormap=:PiYG_8, levels=20)
    Colorbar(fig[1, 2], co)
    fig
end
save("shock_dq.pdf", fig)
