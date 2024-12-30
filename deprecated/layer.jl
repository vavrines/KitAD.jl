using KitBase, Plots
using Base.Threads: @threads
using KitBase.ProgressMeter: @showprogress

function step!(w, prim, h, b, fwL, fhL, fbL, fwR, fhR, fbR, u, v, K, γ, μᵣ, ω, Δs, dt)
    @. w += (fwL - fwR) / Δs
    prim .= conserve_prim(w, γ)

    MH = maxwellian(u, v, prim)
    MB = MH .* K ./ (2.0 * prim[end])
    τ = vhs_collision_time(prim, μᵣ, ω)

    for i in eachindex(u)
        h[i] = (h[i] + (fhL[i] - fhR[i]) / Δs + dt / τ * MH[i]) / (1.0 + dt / τ)
        b[i] = (b[i] + (fbL[i] - fbR[i]) / Δs + dt / τ * MB[i]) / (1.0 + dt / τ)
    end

    return nothing
end

function up!(ks, ctr, face, dt)
    @inbounds @threads for i in 1:ks.ps.nx
        step!(
            ctr[i].w,
            ctr[i].prim,
            ctr[i].h,
            ctr[i].b,
            face[i].fw,
            face[i].fh,
            face[i].fb,
            face[i+1].fw,
            face[i+1].fh,
            face[i+1].fb,
            ks.vs.u,
            ks.vs.v,
            ks.gas.K,
            ks.gas.γ,
            ks.gas.μᵣ,
            ks.gas.ω,
            ks.ps.dx[i],
            dt,
        )
    end

    return nothing
end

begin
    set =
        Setup(; case="layer", space="1d2f2v", maxTime=0.2, boundary=["fix", "fix"], cfl=0.5)
    ps = PSpace1D(-0.5, 0.5, 500, 1)
    vs = VSpace2D(-5.0, 5.0, 28, -5.0, 5.0, 64)
    gas = Gas(; Kn=5e-3, K=1.0)
    fw = function (x, p)
        prim = zeros(4)
        if x <= 0
            prim .= [1.0, 0.0, 1.0, 1.0]
        else
            prim .= [1.0, 0.0, -1.0, 2.0]
        end

        return prim_conserve(prim, ks.gas.γ)
    end
    ff = function (x, p)
        w = fw(x, p)
        prim = conserve_prim(w, gas.γ)
        h = maxwellian(vs.u, vs.v, prim)
        b = energy_maxwellian(h, prim, gas.K)

        return h, b
    end
    bc = function (x, p)
        w = fw(x)
        prim = conserve_prim(w, gas.γ)

        return prim
    end
    ib = IB2F(fw, ff, bc, nothing)
    ks = SolverSet(set, ps, vs, gas, ib)
    ctr, face = init_fvm(ks, ks.ps)
end

τ0 = vhs_collision_time(ctr[1].prim, ks.gas.μᵣ, ks.gas.ω)
tmax = 10τ0#50τ0
t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(tmax ÷ dt)
res = zero(ctr[1].w)

@showprogress for iter in 1:nt
    #reconstruct!(ks, ctr)

    @inbounds @threads for i in 1:ks.ps.nx+1
        flux_kfvs!(
            face[i].fw,
            face[i].fh,
            face[i].fb,
            ctr[i-1].h,
            ctr[i-1].b,
            ctr[i].h,
            ctr[i].b,
            ks.vs.u,
            ks.vs.v,
            ks.vs.weights,
            dt,
            1.0,
        )
    end

    up!(ks, ctr, face, dt)

    #=global t += dt
    if abs(t - τ0) < dt
        @save "sol_t.jld2" ctr face
    elseif abs(t - 10 * τ0) < dt
        @save "sol_10t.jld2" ctr face
    end=#
end
#@save "sol_50t.jld2" ctr face

# field
sol = zeros(ks.ps.nx, 4)
for i in axes(sol, 1)
    sol[i, :] .= ctr[i].prim
    sol[i, end] = 1 / sol[i, end]
end

plot(ks.ps.x[1:ks.ps.nx], sol[:, 3])

# distribution function
fc = (ctr[end÷2].h + ctr[end÷2+1].h) ./ 2
hc = reduce_distribution(fc, vs.weights[:, 1], 2)
plot(ks.vs.v[1, :], hc)
