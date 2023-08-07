isamodel(::Any) = false
isamodel(::Model) = true

"""
    params(m::Model)

Recursively convert any object subtyping `Model` into a named tuple,
keyed on the property names of `m`. The named tuple is possibly nested
because `params` is recursively applied to the property values, which
themselves might subtype `Model`.

For most `Model` objects, properties are synonymous with fields, but
this is not a hard requirement.

    julia> params(EnsembleModel(atom=ConstantClassifier()))
    (atom = (target_type = Bool,),
     weights = Float64[],
     bagging_fraction = 0.8,
     rng_seed = 0,
     n = 100,
     parallel = true,)

"""
params(m) = params(m, Val(isamodel(m)))
params(m, ::Val{false}) = m
function params(m, ::Val{true})
    fields = propertynames(m)
    NamedTuple{fields}(Tuple([params(getproperty(m, field))
                              for field in fields]))
end

"""
    flat_params(t::NamedTuple)

View a nested named tuple `t` as a tree and return, as a Dict, the key subtrees
and the values at the leaves, in the order they appear in the original tuple.

```julia-repl
julia> t = (X = (x = 1, y = 2), Y = 3)
julia> flat_params(t)
LittleDict{...} with 3 entries:
"X__x" => 1
"X__y" => 2
"Y"   => 3
```
"""
function flat_params(parameters::NamedTuple)
    result = LittleDict{String, Any}()
    for key in keys(parameters)
        value = params(getproperty(parameters, key))
        if value isa NamedTuple
            sub_dict = flat_params(value)
            for (sub_key, sub_value) in pairs(sub_dict)
                new_key = string(key, "__", sub_key)
                result[new_key] = sub_value
            end
        else
            result[string(key)] = value
        end
    end
    return result
end
