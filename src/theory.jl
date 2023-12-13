function flux_kfvs(fL, fR, u)
    δ = KB.heaviside(u)
    f = u * (fL * δ + fR * (1.0 - δ))
    return u * f
end

function moments_conserve(f, u, ω)
    return [
        KB.discrete_moments(f, u, ω, 0),
        KB.discrete_moments(f, u, ω, 1),
        KB.discrete_moments(f, u, ω, 2),
    ]
end

function conserve_prim(W, γ)
    return [W[1], W[2] / W[1], 0.5 * W[1] / (γ - 1.0) / (W[3] - 0.5 * W[2]^2 / W[1])]
end
