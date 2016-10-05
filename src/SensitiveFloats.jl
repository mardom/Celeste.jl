module SensitiveFloats

export SensitiveFloat,
       zero_sensitive_float,
       clear!,
       multiply_sfs!,
       add_scaled_sfs!,
       combine_sfs!,
       add_sources_sf!,
       set_hess!

import Base.zero


abstract ParamSet


abstract SensitiveFloat

"""
A function value and its derivative with respect to its arguments.

Attributes:
  v:  The value
  d:  The derivative with respect to each variable in
      P-dimensional VariationalParams for each of S celestial objects
      in a local_P x local_S matrix.
  h:  The second derivative with respect to each variational parameter,
      in the same format as d.  This is used for the full Hessian
      with respect to all the sources.
"""
type SensitiveFloatH{ParamType <: ParamSet, NumType <: Number} <: SensitiveFloat
    # Actually a single value, but an Array to avoid memory allocation
    v::Vector{NumType}

    # local_P x local_S matrix of gradients
    d::Matrix{NumType}

    # h is ordered so that p changes fastest.  For example, the indices
    # of a column of h correspond to the indices of d's stacked columns.
    h::Matrix{NumType} # (local_P * local_S) x (local_P * local_S)
end


type SensitiveFloatD{ParamType <: ParamSet, NumType <: Number} <: SensitiveFloat
    v::Vector{NumType}
    d::Matrix{NumType}
end


type SensitiveFloatV{ParamType <: ParamSet, NumType <: Number} <: SensitiveFloat
    v::Vector{NumType}
end


#########################################################

"""
Set a SensitiveFloat's hessian term, maintaining symmetry.
"""
function set_hess!{ParamType <: ParamSet, NumType <: Number}(
                    sf::SensitiveFloatH{ParamType, NumType},
                    i::Int,
                    j::Int,
                    v::NumType)
    # if i == j, this is still probably faster than branching
    # TODO: make the Hessian have type Hermitian
    sf.h[i, j] = sf.h[j, i] = v
end


function zero{ParamType <: ParamSet, NumType <: Number}(
                    ::Type{SensitiveFloatH{ParamType, NumType}},
                    local_S::Int)
    local_P = length(ParamType)
    v = zeros(NumType, 1)
    d = zeros(NumType, local_P, local_S)
    h = zeros(NumType, local_P * local_S, local_P * local_S)
    SensitiveFloatH{ParamType, NumType}(v, d, h)
end


function zero{ParamType <: ParamSet, NumType <: Number}(
                    ::Type{SensitiveFloatD{ParamType, NumType}},
                    local_S::Int)
    local_P = length(ParamType)
    v = zeros(NumType, 1)
    d = zeros(NumType, local_P, local_S)
    SensitiveFloatD{ParamType, NumType}(v, d)
end


function zero{ParamType <: ParamSet, NumType <: Number}(
                    ::Type{SensitiveFloatV{ParamType, NumType}},
                    local_S::Int)
    local_P = length(ParamType)
    v = zeros(NumType, 1)
    SensitiveFloatV{ParamType, NumType}(v)
end


function clear!{ParamType <: ParamSet, NumType <: Number}(
                sp::SensitiveFloatH{ParamType, NumType})
    fill!(sp.v, zero(NumType))
    fill!(sp.d, zero(NumType))
    fill!(sp.h, zero(NumType))
end


function clear!{ParamType <: ParamSet, NumType <: Number}(
                sp::SensitiveFloatD{ParamType, NumType})
    fill!(sp.v, zero(NumType))
    fill!(sp.d, zero(NumType))
end


function clear!{ParamType <: ParamSet, NumType <: Number}(
                sp::SensitiveFloatV{ParamType, NumType})
    fill!(sp.v, zero(NumType))
end


"""
Factor out the hessian part of combine_sfs!.
"""
function combine_sfs_hessian!{ParamType <: ParamSet,
                 T1 <: Number, T2 <: Number, T3 <: Number}(
            sf1::SensitiveFloatH{ParamType, T1},
            sf2::SensitiveFloatH{ParamType, T1},
            sf_result::SensitiveFloatH{ParamType, T1},
            g_d::Vector{T2}, g_h::Matrix{T3})
    @assert g_h[1, 2] == g_h[2, 1]

    p1, p2 = size(sf_result.h)
    @assert size(sf_result.h) == size(sf1.h) == size(sf2.h)
    @assert p1 == p2 == prod(size(sf1.d)) == prod(size(sf2.d))
    for ind2 = 1:p2
        sf11_factor = g_h[1, 1] * sf1.d[ind2] + g_h[1, 2] * sf2.d[ind2]
        sf21_factor = g_h[1, 2] * sf1.d[ind2] + g_h[2, 2] * sf2.d[ind2]

        @inbounds for ind1 = 1:ind2
            sf_result.h[ind1, ind2] =
                g_d[1] * sf1.h[ind1, ind2] +
                g_d[2] * sf2.h[ind1, ind2] +
                sf11_factor * sf1.d[ind1] +
                sf21_factor * sf2.d[ind1]
            sf_result.h[ind2, ind1] = sf_result.h[ind1, ind2]
        end
    end

    true # Set definite return type
end


"""
Updates sf_result in place with g(sf1, sf2), where
g_d = (g_1, g_2) is the gradient of g and
g_h = (g_11, g_12; g_12, g_22) is the hessian of g,
each evaluated at (sf1, sf2).

The result is stored in sf_result.  The order is done in such a way that
it can overwrite sf1 or sf2 and still be accurate.
"""
function combine_sfs!{ParamType <: ParamSet,
                      T1 <: Number, T2 <: Number, T3 <: Number}(
        sf1::SensitiveFloatH{ParamType, T1},
        sf2::SensitiveFloatH{ParamType, T1},
        sf_result::SensitiveFloatH{ParamType, T1},
        v::T1, g_d::Vector{T2}, g_h::Matrix{T3})
    # You have to do this in the right order to not overwrite needed terms.
    combine_sfs_hessian!(sf1, sf2, sf_result, g_d, g_h)

    combine_sfs!(sf1, sf2, sf_results, v g_d, g_h)

    sf_result.d[:] = sf1.d
    n = length(sf_result.d)
    LinAlg.BLAS.scal!(n, g_d[1], sf_result.d, 1)
    LinAlg.BLAS.axpy!(g_d[2], sf2.d, sf_result.d)

    sf_result.v[1] = v
end


function combine_sfs!{ParamType <: ParamSet,
                      T1 <: Number, T2 <: Number, T3 <: Number}(
        sf1::SensitiveFloatH{ParamType, T1},
        sf2::SensitiveFloatH{ParamType, T1},
        sf_result::SensitiveFloatH{ParamType, T1},
        v::T1, g_d::Vector{T2}, g_h::Matrix{T3};
        calculate_hessian::Bool=true)
    # You have to do this in the right order to not overwrite needed terms.
    if calculate_hessian
        combine_sfs_hessian!(sf1, sf2, sf_result, g_d, g_h)
    end

    sf_result.d[:] = sf1.d
    n = length(sf_result.d)
    LinAlg.BLAS.scal!(n, g_d[1], sf_result.d, 1)
    LinAlg.BLAS.axpy!(g_d[2], sf2.d, sf_result.d)

    sf_result.v[1] = v
end


"""
Updates sf1 in place with g(sf1, sf2), where
g_d = (g_1, g_2) is the gradient of g and
g_h = (g_11, g_12; g_12, g_22) is the hessian of g,
each evaluated at (sf1, sf2).

The result is stored in sf1.
"""
function combine_sfs!{ParamType <: ParamSet,
                      T1 <: Number, T2 <: Number, T3 <: Number}(
        sf1::SensitiveFloat{ParamType, T1},
        sf2::SensitiveFloat{ParamType, T1},
        v::T1, g_d::Vector{T2}, g_h::Matrix{T3};
        calculate_hessian::Bool=true)

    combine_sfs!(sf1, sf2, sf1, v, g_d, g_h, calculate_hessian=calculate_hessian)

    true # Set definite return type
end

# Decalare outside to avoid allocating memory.
const multiply_sfs_hess = Float64[0 1; 1 0]

"""
Updates sf1 in place with sf1 * sf2.
"""
function multiply_sfs!{ParamType <: ParamSet, NumType <: Number}(
        sf1::SensitiveFloat{ParamType, NumType},
        sf2::SensitiveFloat{ParamType, NumType},
        calculate_hessian::Bool=true)

    v = sf1.v[1] * sf2.v[1]
    g_d = NumType[sf2.v[1], sf1.v[1]]
    #const g_h = NumType[0 1; 1 0]

    combine_sfs!(sf1, sf2, v, g_d, multiply_sfs_hess,
                             calculate_hessian=calculate_hessian)

    true # Set definite return type
end


"""
Update sf1 in place with (sf1 + scale * sf2).
"""
function add_scaled_sfs!{ParamType <: ParamSet, NumType <: Number}(
        sf1::SensitiveFloat{ParamType, NumType},
        sf2::SensitiveFloat{ParamType, NumType},
        scale::AbstractFloat, calculate_hessian::Bool)

    sf1.v[1] += scale * sf2.v[1]

    LinAlg.BLAS.axpy!(scale, sf2.d, sf1.d)

    if calculate_hessian
        p1, p2 = size(sf1.h)
        @assert (p1, p2) == size(sf2.h)
        @inbounds for ind2=1:p2, ind1=1:ind2
            sf1.h[ind1, ind2] += scale * sf2.h[ind1, ind2]
            sf1.h[ind2, ind1] = sf1.h[ind1, ind2]
        end
    end

    true # Set definite return type
end


"""
Adds sf2_s to sf1, where sf1 is sensitive to multiple sources and sf2_s is only
sensitive to source s.
"""
function add_sources_sf!{ParamType <: ParamSet, NumType <: Number}(
        sf_all::SensitiveFloat{ParamType, NumType},
        sf_s::SensitiveFloat{ParamType, NumType},
        s::Int, calculate_hessian::Bool)

    # TODO: This line, too, allocates a lot of memory.    Why?
    sf_all.v[1] = sf_all.v[1] + sf_s.v[1]

    # TODO: time consuming **************
    P = length(ParamType)
    Ph = size(sf_all.h)[1]

    @assert Ph == prod(size(sf_all.d))
    @assert Ph == size(sf_all.h)[2]
    @assert Ph >= P * s
    @assert size(sf_s.d) == (P, 1)
    @assert size(sf_s.h) == (P, P)

    @inbounds for s_ind1 in 1:P
        s_all_ind1 = P * (s - 1) + s_ind1
        sf_all.d[s_all_ind1] = sf_all.d[s_all_ind1] + sf_s.d[s_ind1]
        if calculate_hessian
            @inbounds for s_ind2 in 1:s_ind1
                s_all_ind2 = P * (s - 1) + s_ind2
                sf_all.h[s_all_ind2, s_all_ind1] =
                    sf_all.h[s_all_ind2, s_all_ind1] + sf_s.h[s_ind2, s_ind1]
                # TODO: move outside the loop?
                sf_all.h[s_all_ind1, s_all_ind2] = sf_all.h[s_all_ind2, s_all_ind1]
            end
        end
    end

    true # Set definite return type
end

end
