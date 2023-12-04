"""
KitAD.jl: The lightweight module of automatic differentiations in Kinetic.jl

Copyright (c) 2023 Tianbai Xiao <tianbaixiao@gmail.com>
"""

module KitAD

export KA

import KitBase as KB

include("theory.jl")

const KA = KitAD

end
