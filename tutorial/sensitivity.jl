using OrdinaryDiffEq, SciMLSensitivity

function rhs(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + u[1] * u[2]
end

u0 = [1.0, 1.0]
p0 = [1.5, 1.0, 3.0]
tspan = (0.0, 1.0)
tsteps = tspan[1]:0.01:tspan[2]

#--- direct forward sensitivity ---#
prob0 = ODEForwardSensitivityProblem(f, u0, tspan, p0)
sol0 = solve(prob0, Midpoint(); saveat = tsteps)
x, dp = extract_local_sensitivities(sol0)

#--- adjoint ---#
prob = ODEProblem(f, u0, tspan, p0)
sol = solve(prob, Tsit5(); saveat = tsteps)

# L = u + v
g(u, p, t) = sum(u)
function dg(out, u, p, t)
    out[1] = 1
    out[2] = 1
end

res = adjoint_sensitivities(sol, Tsit5(); dgdu_continuous = dg, g = g, saveat = tsteps)

#--- automatic differentiation ---#
# quadrature function
using QuadGK, ForwardDiff

function loss(p)
    prob1 = remake(prob, p = p)
    sol = solve(prob1, Tsit5())
    res = quadgk((t) -> (sum(sol(t))), tspan[1], tspan[2])[1]
end

ForwardDiff.gradient(loss, p0)

# hand-written quadrature
using Zygote

function loss1(p)
    prob1 = remake(prob, p = p)
    sol = solve(prob1, Tsit5(); saveat = tsteps) |> Array
    weights = (tspan[2] - tspan[1]) / length(tsteps)
    return sum(sol) .* weights
end

Zygote.pullback(x -> loss1(x), p0)[2](1)[1]

# energy functional: L = (u^2 + v^2)/2
g(u, p, t) = (sum(u) .^ 2) ./ 2
function dg(out, u, p, t)
    out[1] = u[1] + u[2]
    out[2] = u[1] + u[2]
end
