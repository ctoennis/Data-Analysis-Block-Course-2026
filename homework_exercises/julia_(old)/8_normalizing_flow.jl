### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ a052ebee-b85a-41e9-aa46-1c5fc050b0c9
begin
    import Pkg
    # activate a temporary environment
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="HighEnergyTools", url="https://github.com/RUB-EP1/HighEnergyTools.jl"),
		Pkg.PackageSpec(name="Enzyme"),
		Pkg.PackageSpec(name="Optimisers"),
    	Pkg.PackageSpec(name="ConcreteStructs"),
		Pkg.PackageSpec(name="MLUtils"),
		Pkg.PackageSpec(name="Lux"),
		Pkg.PackageSpec(name="Reactant"),
		Pkg.PackageSpec(name="Printf"),
	])
    using HighEnergyTools, Plots, HighEnergyTools.Statistics, HighEnergyTools.Random, Lux, Reactant, Enzyme, MLUtils, ConcreteStructs, Optimisers, Printf
end

# ╔═╡ 72a7ae28-6528-4bfa-8626-3f4a20c81d64
md"""
# [Normalizing Flows for Density Estimation](@id RealNVP-Tutorial)

- This tutorial demonstrates how to use Lux to train a
- [RealNVP](https://arxiv.org/abs/1605.08803). This is based on the
- [RealNVP implementation in MLX](https://github.com/ml-explore/mlx-examples/blob/main/normalizing_flow/).
"""

# ╔═╡ 787020db-eb10-4da6-b2a6-b3a90b6e9a31
theme(:boxed, markersize=1, markerstrokewidth=0, c=6, size=(700,600))

# ╔═╡ f8f9d67a-3e8f-431a-ab62-d8531e699027
begin
	const xdev = reactant_device(; force=true)
	const cdev = cpu_device()
end

# ╔═╡ c00ecf6c-300c-4047-b95c-83ad7a03040d
function make_moons(
    rng::AbstractRNG,
    ::Type{T},
    n_samples::Int=100;
    noise::Union{Nothing,AbstractFloat}=nothing,
) where {T}
    n_moons = n_samples ÷ 2
    t_min, t_max = T(0), T(π)
    t_inner = rand(rng, T, n_moons) * (t_max - t_min) .+ t_min
    t_outer = rand(rng, T, n_moons) * (t_max - t_min) .+ t_min
    outer_circ_x = cos.(t_outer)
    outer_circ_y = sin.(t_outer) .+ T(1)
    inner_circ_x = 1 .- cos.(t_inner)
    inner_circ_y = 1 .- sin.(t_inner) .- T(1)

    data = [outer_circ_x outer_circ_y; inner_circ_x inner_circ_y]
    z = permutedims(data, (2, 1))
    noise !== nothing && (z .+= T(noise) * randn(rng, T, size(z)))
    return z
end

# ╔═╡ 0a3888c6-4cd1-4f55-8822-4c306c5ff029
md"""
## Let's visualize the dataset
"""

# ╔═╡ babd51b9-df9d-46b2-8baf-d753e39349a7
sample_z = make_moons(Random.default_rng(), Float32, 10_000; noise=0.1)

# ╔═╡ c0285803-1263-4ddf-a554-2b73d44b0913
begin
	plot(aspect_ratio=1)
	scatter!(sample_z[1, :], sample_z[2, :]; markersize=2, xlabel="x", ylabel="y")
end

# ╔═╡ 27943aff-320d-46f9-89d3-797e014fa4f3
md"""
## Bijectors Implementation
"""

# ╔═╡ 92c19da0-c379-423a-bee2-e0d87aa4362a
md"""
## Model Definition
"""

# ╔═╡ 18eba242-c252-424b-a230-85a99d4d1fe6
md"""
## Helper Functions
"""

# ╔═╡ 2c8e2684-5548-4ea2-b300-fa72a9503863
dsum(x; dims) = dropdims(sum(x; dims); dims)

# ╔═╡ 4a2a3f62-4638-4ac9-8d60-65b32f73ba32
begin
	abstract type AbstractBijector end
	
	@concrete struct AffineBijector <: AbstractBijector
	    shift <: AbstractArray
	    log_scale <: AbstractArray
	end
	
	function AffineBijector(shift_and_log_scale::AbstractArray{T,N}) where {T,N}
	    n = size(shift_and_log_scale, 1) ÷ 2
	    idxs = ntuple(Returns(Colon()), N - 1)
	    return AffineBijector(
	        shift_and_log_scale[1:n, idxs...], shift_and_log_scale[(n + 1):end, idxs...]
	    )
	end
	
	function forward_and_log_det(bj::AffineBijector, x::AbstractArray)
	    y = x .* exp.(bj.log_scale) .+ bj.shift
	    return y, bj.log_scale
	end
	
	function inverse_and_log_det(bj::AffineBijector, y::AbstractArray)
	    x = (y .- bj.shift) ./ exp.(bj.log_scale)
	    return x, -bj.log_scale
	end
	
	@concrete struct MaskedCoupling <: AbstractBijector
	    mask <: AbstractArray
	    conditioner
	    bijector
	end
	
	function apply_mask(bj::MaskedCoupling, x::AbstractArray, fn::F) where {F}
	    x_masked = x .* (1 .- bj.mask)
	    bijector_params = bj.conditioner(x_masked)
	    y, log_det = fn(bijector_params)
	    log_det = log_det .* bj.mask
	    y = ifelse.(bj.mask, y, x)
	    return y, dsum(log_det; dims=Tuple(collect(1:(ndims(x) - 1))))
	end
	
	function forward_and_log_det(bj::MaskedCoupling, x::AbstractArray)
	    return apply_mask(bj, x, params -> forward_and_log_det(bj.bijector(params), x))
	end
	
	function inverse_and_log_det(bj::MaskedCoupling, y::AbstractArray)
	    return apply_mask(bj, y, params -> inverse_and_log_det(bj.bijector(params), y))
	end
end

# ╔═╡ 4bc52d63-4808-4fb2-ba2b-396a8c54a7f5
begin
	function MLP(in_dims::Int, hidden_dims::Int, out_dims::Int, n_layers::Int; activation=gelu)
	    return Chain(
	        Dense(in_dims => hidden_dims, activation),
	        [Dense(hidden_dims => hidden_dims, activation) for _ in 1:(n_layers - 1)]...,
	        Dense(hidden_dims => out_dims),
	    )
	end
	
	@concrete struct RealNVP <: AbstractLuxContainerLayer{(:conditioners,)}
	    conditioners
	    dist_dims::Int
	    n_transforms::Int
	end
	
	const StatefulRealNVP{M} = StatefulLuxLayer{M,<:RealNVP}
	
	function Lux.initialstates(rng::AbstractRNG, l::RealNVP)
	    mask_list = Vector{Bool}[
	        collect(1:(l.dist_dims)) .% 2 .== i % 2 for i in 1:(l.n_transforms)
	    ]
	    return (; mask_list, conditioners=Lux.initialstates(rng, l.conditioners))
	end
	
	function RealNVP(; n_transforms::Int, dist_dims::Int, hidden_dims::Int, n_layers::Int)
	    conditioners = [
	        MLP(dist_dims, hidden_dims, 2 * dist_dims, n_layers; activation=gelu) for
	        _ in 1:n_transforms
	    ]
	    conditioners = NamedTuple{ntuple(Base.Fix1(Symbol, :conditioners_), n_transforms)}(
	        Tuple(conditioners)
	    )
	    return RealNVP(conditioners, dist_dims, n_transforms)
	end
	
	log_prob(x::AbstractArray{T}) where {T} = -T(0.5 * log(2π)) .- T(0.5) .* abs2.(x)
	
	function log_prob(l::StatefulRealNVP, x::AbstractArray{T}) where {T}
	    smodels = [
	        StatefulLuxLayer{true}(conditioner, l.ps.conditioners[i], l.st.conditioners[i]) for
	        (i, conditioner) in enumerate(l.model.conditioners)
	    ]
	
	    lprob = zeros_like(x, size(x, ndims(x)))
	    for (mask, conditioner) in Iterators.reverse(zip(l.st.mask_list, smodels))
	        bj = MaskedCoupling(mask, conditioner, AffineBijector)
	        x, log_det = inverse_and_log_det(bj, x)
	        lprob += log_det
	    end
	    lprob += dsum(log_prob(x); dims=Tuple(collect(1:(ndims(x) - 1))))
	
	    conditioners = NamedTuple{
	        ntuple(Base.Fix1(Symbol, :conditioners_), l.model.n_transforms)
	    }(
	        Tuple([smodel.st for smodel in smodels])
	    )
	    l.st = merge(l.st, (; conditioners))
	
	    return lprob
	end
end

# ╔═╡ 813d7bb7-93fe-4394-9d2c-32f0f15172ff
function loss_function(model, ps, st, x)
	smodel = StatefulLuxLayer{true}(model, ps, st)
	lprob = log_prob(smodel, x)
	return -mean(lprob), smodel.st, (;)
end

# ╔═╡ fffd93e8-668a-4de3-a1a4-f82ccf2cf46d
md"""
## Training the Model
"""

# ╔═╡ 2103bac4-d978-4988-839a-72324db495b0
rng = let
	_rng = Random.default_rng()
    Random.seed!(_rng, 0)
end

# ╔═╡ a8f1f9f3-5eec-4bda-bfd3-cdd477fec880
dataloader = let
    n_train_samples = 100_000
	# 
	sample = make_moons(rng, Float32, n_train_samples; noise = 0.06)
	DL = DataLoader(sample; batchsize = 128, shuffle=true, partial=false)
	# 
    DL |> xdev |> Iterators.cycle
end

# ╔═╡ 8e9d5fb8-6322-4a3b-8b10-03db116039e5
const pure_model = RealNVP(; 
	n_transforms = 6,
	dist_dims=2,
	hidden_dims = 16,
	n_layers = 4);

# ╔═╡ b12d9c6c-c008-4cf2-9094-ed202e07f564
trained_model = let
	# 
    maxiters = 10_000
    lr = 0.0004
		
    ps, st = xdev(Lux.setup(rng, pure_model))
    opt = Adam(lr)

    train_state = Training.TrainState(pure_model, ps, st, opt)
    @info "Total Trainable Parameters: $(Lux.parameterlength(ps))"

    total_samples = 0
    start_time = time()

    for (iter, x) in enumerate(dataloader)
        total_samples += size(x, ndims(x))
        (_, loss, _, train_state) = Training.single_train_step!(
            AutoEnzyme(), loss_function, x, train_state; return_gradients=Val(false)
        )

        isnan(loss) && error("NaN loss encountered in iter $(iter)!")

        if iter == 1 || iter == maxiters || iter % 1000 == 0
            throughput = total_samples / (time() - start_time)
            @info "Iter: [$(iter)/$(maxiters)]    Training Loss: $(round(collect(loss)[1], digits=3))    Throughput: $(round(throughput; digits=1)) samples"
        end
        iter ≥ maxiters && break
    end

    StatefulLuxLayer{true}(
        pure_model, train_state.parameters, Lux.testmode(train_state.states)
    )
end

# ╔═╡ bc5e42c5-3233-4c9b-a24c-546ca2aa3d74
md"""
## Explore the first layer
"""

# ╔═╡ 44ecff9f-931e-4042-9385-be27741150dd
let
	
	Lux.apply(
			trained_model.model.conditioners.conditioners_1,
			  xdev(rand(2,30)),
			  trained_model.ps.conditioners.conditioners_1,
			  Lux.testmode(trained_model.st.conditioners.conditioners_1))[1] |> cdev
end

# ╔═╡ 531395eb-145b-4d7d-8e95-3909db40e8f6
md"""
## Visualizing the Results
"""

# ╔═╡ 0a7e9689-d39f-436a-bf0d-b0b0793e3784
function sample(
    rng::AbstractRNG,
    ::Type{T},
    d::StatefulRealNVP,
    nsamples::Int,
    nsteps::Int=length(d.model.conditioners),
) where {T}
    @assert 1 ≤ nsteps ≤ length(d.model.conditioners)

    smodels = [
        StatefulLuxLayer{true}(conditioner, d.ps.conditioners[i], d.st.conditioners[i]) for
        (i, conditioner) in enumerate(d.model.conditioners)
    ]

    x = randn(rng, T, d.model.dist_dims, nsamples)
    for (i, (mask, conditioner)) in enumerate(zip(d.st.mask_list, smodels))
        x, _ = forward_and_log_det(MaskedCoupling(mask, conditioner, AffineBijector), x)
        i ≥ nsteps && break
    end
    return x
end

# ╔═╡ d288e391-5443-4393-b2e5-7b9ae7ba0717
begin
	z_stages = Matrix{Float32}[]
	for i in 1:(trained_model.model.n_transforms)
	    z = @jit sample(Random.default_rng(), Float32, trained_model, 10_000, i)
	    push!(z_stages, Array(z))
	end
end

# ╔═╡ 3eeda3c3-9d08-43ae-971d-8804708baf2e
let
	plot(layout=grid(2,3), size=(1200, 800))
    for (sp, z) in enumerate(z_stages)
        scatter!(z[1, :], z[2, :]; markersize=2, markerstrokewidth=0, sp)
    end
    plot!()
end

# ╔═╡ Cell order:
# ╟─72a7ae28-6528-4bfa-8626-3f4a20c81d64
# ╠═a052ebee-b85a-41e9-aa46-1c5fc050b0c9
# ╠═787020db-eb10-4da6-b2a6-b3a90b6e9a31
# ╠═f8f9d67a-3e8f-431a-ab62-d8531e699027
# ╠═c00ecf6c-300c-4047-b95c-83ad7a03040d
# ╟─0a3888c6-4cd1-4f55-8822-4c306c5ff029
# ╠═babd51b9-df9d-46b2-8baf-d753e39349a7
# ╠═c0285803-1263-4ddf-a554-2b73d44b0913
# ╟─27943aff-320d-46f9-89d3-797e014fa4f3
# ╠═4a2a3f62-4638-4ac9-8d60-65b32f73ba32
# ╟─92c19da0-c379-423a-bee2-e0d87aa4362a
# ╠═4bc52d63-4808-4fb2-ba2b-396a8c54a7f5
# ╟─18eba242-c252-424b-a230-85a99d4d1fe6
# ╠═2c8e2684-5548-4ea2-b300-fa72a9503863
# ╠═813d7bb7-93fe-4394-9d2c-32f0f15172ff
# ╟─fffd93e8-668a-4de3-a1a4-f82ccf2cf46d
# ╠═2103bac4-d978-4988-839a-72324db495b0
# ╠═a8f1f9f3-5eec-4bda-bfd3-cdd477fec880
# ╠═8e9d5fb8-6322-4a3b-8b10-03db116039e5
# ╠═b12d9c6c-c008-4cf2-9094-ed202e07f564
# ╟─bc5e42c5-3233-4c9b-a24c-546ca2aa3d74
# ╠═44ecff9f-931e-4042-9385-be27741150dd
# ╟─531395eb-145b-4d7d-8e95-3909db40e8f6
# ╠═0a7e9689-d39f-436a-bf0d-b0b0793e3784
# ╠═d288e391-5443-4393-b2e5-7b9ae7ba0717
# ╠═3eeda3c3-9d08-43ae-971d-8804708baf2e
