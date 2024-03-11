using OrdinaryDiffEq, Solaris, SciMLSensitivity
using Optimisers: Adam
using Optim: LBFGS
using Optimization: AutoZygote

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    du[2] = -δ * y + γ * x * y
end

u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0
p0 = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(lotka_volterra!, u0, tspan, p0)

function loss(p)
    sol = solve(
        prob,
        Tsit5(),
        p = p,
        saveat = tsteps,
        sensealg = InterpolatingAdjoint(autojacvec = EnzymeVJP()),
    )
    loss = sum(abs2, sol .- 1)

    return loss
end

cb = function (p, l)
    display(l)
    return false
end

res = sci_train(loss, p0, Adam(); cb = cb, iters = 1000, ad = AutoZygote())
res = sci_train(loss, res.u, LBFGS(); cb = cb, iters = 1000, ad = AutoZygote())

@show res.u, loss(res.u)

function loss1(p; vjp)
    sol = solve(
        prob,
        Tsit5(),
        p = p,
        saveat = tsteps,
        sensealg = InterpolatingAdjoint(autojacvec = vjp),
    )
    loss = sum(abs2, sol .- 1)

    return loss
end

using Zygote, BenchmarkTools

# preferred
@btime Zygote.pullback(x -> loss1(x; vjp = EnzymeVJP()), p0)[2](1)[1]

# undesirable
@btime Zygote.pullback(x -> loss1(x; vjp = ReverseDiffVJP()), p0)[2](1)[1]
@btime Zygote.pullback(x -> loss1(x; vjp = TrackerVJP()), p0)[2](1)[1]
@btime Zygote.pullback(x -> loss1(x; vjp = ZygoteVJP()), p0)[2](1)[1]

# also efficient
@btime Zygote.pullback(x -> loss1(x; vjp = true), p0)[2](1)[1]
@btime Zygote.pullback(x -> loss1(x; vjp = false), p0)[2](1)[1]

function loss2(p; vjp)
    sol = solve(prob, Tsit5(), p = p, saveat = tsteps, sensealg = ForwardDiffSensitivity())
    loss = sum(abs2, sol .- 1)

    return loss
end

# more efficient since the dimension is low
@btime Zygote.pullback(x -> loss2(x), p0)[2](1)[1]

# discretize-then-optimize approach
function loss3(p)
    sol = solve(prob, Tsit5(), p = p, saveat = tsteps, sensealg = ReverseDiffAdjoint())
    loss = sum(abs2, sol .- 1)

    return loss
end

Zygote.pullback(x -> loss3(x), p0)[2](1)[1]
sci_train(loss3, p0, Adam(); cb = cb, iters = 1000, ad = AutoZygote())
