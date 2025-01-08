using KitBase, Plots
using KitBase.ProgressMeter: @showprogress

set = Setup(; case="shock", space="1d2f1v", collision="bgk", maxTime=50)
ps = PSpace1D(-25, 25, 50, 1)
vs = VSpace1D(-8, 8, 36)
gas = Gas(; Kn=1.0, Ma=3.0, K=2.0)
ib = IB2F(KB.config_ib(set, ps, vs, gas)...)

ks = SolverSet(set, ps, vs, gas, ib)
ctr, face = init_fvm(ks, :dynamic_array)

t = 0.0
dt = timestep(ks, ctr, t)
nt = ks.set.maxTime ÷ dt |> Int
res = zeros(3)

@showprogress for iter in 1:nt
    evolve!(ks, ctr, face, dt)
    update!(ks, ctr, face, dt, res)
end

plot(ks, ctr)

using KitBase.JLD2
cd(@__DIR__)
@save "bgk.jld2" ctr
