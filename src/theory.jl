function conserve_prim(W, γ)
    return [W[1], W[2] / W[1], 0.5 * W[1] / (γ - 1.0) / (W[3] - 0.5 * W[2]^2 / W[1])]
end

function maxwellian(u, prim)
    M = zero(u)
    for i in eachindex(M)
        M[i] = KB.maxwellian(u[i], prim[1], prim[2], prim[end])
    end

    return M
end

function energy_maxwellian(h, λ, K)
    b = zero(h)
    for i in eachindex(b)
        b[i] = h[i] * K / (2.0 * λ)
    end

    return b
end

function discrete_moments(f, u, ω, n)
    prods = zero(f)
    for i in eachindex(prods)
        prods[i] = f[i] * u[i]^n * ω[i]
    end

    return sum(prods)
end

function moments_conserve(f, u, ω)
    return [
        discrete_moments(f, u, ω, 0),
        discrete_moments(f, u, ω, 1),
        0.5 * discrete_moments(f, u, ω, 2),
    ]
end

function moments_conserve(h, b, u, ω)
    return [
        KB.discrete_moments(h, u, ω, 0),
        KB.discrete_moments(h, u, ω, 1),
        0.5 * (KB.discrete_moments(h, u, ω, 2) + KB.discrete_moments(b, u, ω, 0)),
    ]
end

function flux_kfvs(fL, fR, u)
    δ = KB.heaviside(u)
    f = u * (fL * δ + fR * (1.0 - δ))
    return u * f
end
