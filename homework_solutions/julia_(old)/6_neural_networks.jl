### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 7a20452e-6600-11f0-2e96-e195c74cdb7e
begin
	import Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="UnROOT"),
		Pkg.PackageSpec(name="DataFrames"),
		Pkg.PackageSpec(name="Flux"),
		Pkg.PackageSpec(name="FHist"),
		Pkg.PackageSpec(name="Random"),
		Pkg.PackageSpec(name="StatsBase"),
    ])
    using UnROOT, Plots, DataFrames, Flux, FHist, Random, StatsBase
end

# ╔═╡ 7f6f1196-3161-4219-b3a6-6eac0fd12fe9
begin
	# get training, validation and test samples
    fbkg = ROOTFile("background.root")
    fsig = ROOTFile("signal.root")
	ftest = ROOTFile("test.root")
    tbkg = LazyTree(fbkg, "t")
	tsig = LazyTree(fsig, "t")
	ttest = LazyTree(ftest, "t")
	# convert root trees to dataframes
	dfbkg = DataFrame(tbkg)
	dfsig = DataFrame(tsig)
	dftest = DataFrame(ttest)
	# get the names of variables for training the BDT	
	feature_names = names(dfsig)
	# if you'd like to chose a subset...
	#feature_names = ["Lc_PT", "Lc_OWNPVVDRHO", "pi_PID_K", "K_PT", "Lc_DOCA_23", "Lc_DOCA_12", "Lc_DOCA_13"]
end

# ╔═╡ c53cd813-e8be-4f97-b4d6-1f445e0df061
begin
	# apply the filter to all variables
	dfbkgf = filter(row -> all(x -> !(x isa Number && isnan(x)), row), dfbkg)
	dfsigf = filter(row -> all(x -> !(x isa Number && isnan(x)), row), dfsig)
	dftestf = filter(row -> all(x -> !(x isa Number && isnan(x)), row), dftest)
end

# ╔═╡ 3e3a86e6-d5fa-4186-93be-8ef6ab93e4b9
log_transform = ["K_OWNPVIPCHI2",
"pi_PT",
"Xi_PT",
"Lc_OWNPVVDRHO",
"Lc_OWNPVVDZ",
"pi_OWNPVIPCHI2",
"Lc_OWNPVFDCHI2",
"Xi_OWNPVIP",
"K_PT",
"Lc_PT"
]

# ╔═╡ b10c90b4-1acb-4a40-9b36-5c7fc80a5329
function split_df(df, frac_train=0.8)
    n = nrow(df)
    idx = randperm(n)
    n_train = round(Int, frac_train * n)
    train_idx = idx[1:n_train]
    val_idx = idx[n_train+1:end]
    return df[train_idx, :], df[val_idx, :]
end

# ╔═╡ f63e1a46-e598-4172-a330-30fdef152e9b
tsig_train, tsig_val, tbkg_train, tbkg_val, t_test = let	
    local_sig = copy(dfsigf)
	local_bkg = copy(dfbkgf)
	local_test = copy(dftest)
	# log transformations
    for var in feature_names
		if var in log_transform
			transform!(local_sig, Symbol(var) => ByRow(log10) => Symbol(var))
			transform!(local_bkg, Symbol(var) => ByRow(log10) => Symbol(var))
			transform!(local_test, Symbol(var) => ByRow(log10) => Symbol(var))
		end
		# trim data to common support
        support_min = max(minimum(local_bkg[!,var]), minimum(local_sig[!,var]))
        support_max = min(maximum(local_bkg[!,var]), maximum(local_sig[!,var]))
        local_bkg = filter(row -> row[var] > support_min && row[var] < support_max, local_bkg)
        local_sig = filter(row -> row[var] > support_min && row[var] < support_max, local_sig)
    end
	# add labels for training and validation data
	local_sig.label .= true  # or "signal"
    local_bkg.label .= false  # or "background"
	# split into validation and training samples
	local_dfsigf_train, local_dfsigf_val = split_df(local_sig, 0.8)
    local_dfbkgf_train, local_dfbkgf_val = split_df(local_bkg, 0.8)
    (local_dfsigf_train, local_dfsigf_val, local_dfbkgf_train, local_dfbkgf_val, local_test)
end

# ╔═╡ 99b776a0-38cc-4f34-a540-2e5e150c3bfb
function flux_trans(df, v)	
	return Float32.(Base.permutedims(Matrix(df[:,v])))
end

# ╔═╡ 6e1d9c8a-3f01-4c8b-9568-3ff8adba497b
begin
	t_train = vcat(tsig_train,tbkg_train)
	t_val = vcat(tsig_val,tbkg_val)	
	
	# convert Dataframe to Matrix, and transpose it for Flux
	# for debugging, it helps to use size(...) to print dimensions of objects
	#size(Base.permutedims(Matrix(tsig_train[:,feature_names])))	
	
	# note that size(...) returns a tuple
	n_features = size(feature_names)[1]
	
	tx_train = flux_trans(t_train,feature_names)
	tx_val = flux_trans(t_val,feature_names)
	ty_train = flux_trans(t_train,["label"])
	ty_val = flux_trans(t_val,["label"])
end

# ╔═╡ f4c36064-3968-44be-8744-49e4c81415c5
md"""
## Exercise 1: One does not simply train a neural network...

Try yourself to setup a "simple" deep NN with a few dense layers that does the classification task.

Hint: define the model, the setup, the array of losses as global variables, and train for a small number of epochs; if you re-evaluate the training cell (Shift+Enter), the next epochs will be trained. In this way, you get a fast response if your setup has the chance to converge to something useful. For the plot to update correctly, you also need to re-evaluate the plot cell. You can also change the number of epochs to a larger number, once you're happy with the setup.

Reducing complexity will also help to get started, for instance by reducing the number of input features.

Learn about NN architectures in the flux documentation. For example about [layers](https://fluxml.ai/Flux.jl/stable/reference/models/layers/), [activation functions](https://fluxml.ai/Flux.jl/stable/reference/models/activation/), [optimisers](https://fluxml.ai/Optimisers.jl/dev/api/) or [losses](https://fluxml.ai/Flux.jl/stable/reference/models/losses/).

You may also want to ask your trusted AI chatbot for recommendations on the architecture.
"""

# ╔═╡ 383a0230-5752-40dd-9374-e7ccc11c77ff
begin
	
	cl_model = Chain(Dense(n_features, 8, rrelu),
					 Dense(8, 1, sigmoid), 
					)
	copts = Flux.setup(Flux.Adam(0.003), cl_model)
	# to keep track of losses
	t_losses = []; v_losses = []
end

# ╔═╡ 933c6c00-1766-463a-ba1e-1bc8a08d65cd
let		
	for epoch in 1:10000
		t_loss, grads = Flux.withgradient(cl_model) do m
			# Evaluate model and loss inside gradient context:
			y_hat = m(tx_train)
			# binary cross-entropy is the same as logloss (that we used for the gradient boosting exercise)
			Flux.Losses.binarycrossentropy(y_hat, ty_train)
		end
		Flux.update!(copts, cl_model, grads[1])
		v_loss = Flux.Losses.binarycrossentropy(cl_model(tx_val), ty_val)
		push!(t_losses, t_loss)
		push!(v_losses, v_loss)
	end
end

# ╔═╡ 66c0d418-714e-4523-b608-4f5f2d163985
let	
    # change the starting index of the array to zoom into later epochs (or use ylims)
	plot(t_losses[500:end], label="Train Loss", lw=2)
	plot!(v_losses[500:end], label="Val Loss", lw=2, title="Loss Curve", xlabel="Epoch", ylabel="BCE")
end

# ╔═╡ b7d0ee07-a929-4d18-97f8-edd0dc6f0206
let
	# add the NN response to the dataframes
	tsig_train.dnn = cl_model(flux_trans(tsig_train, feature_names))[1,:]
	tbkg_train.dnn = cl_model(flux_trans(tbkg_train, feature_names))[1,:]
	tsig_val.dnn = cl_model(flux_trans(tsig_val, feature_names))[1,:]
	tbkg_val.dnn = cl_model(flux_trans(tbkg_val, feature_names))[1,:]
end

# ╔═╡ 28c36329-e978-485d-8d13-3079a777ce9c
let
	# the same plot we know from the boosting exercise: apply trained network to signal and background training and validation samples.
	plotrange = range(0,1,51)
	hst = Hist1D(tsig_train[:,"dnn"],binedges=plotrange)
	hsv = Hist1D(tsig_val[:,"dnn"],binedges=plotrange)
	hbt = Hist1D(tbkg_train[:,"dnn"],binedges=plotrange)
	hbv = Hist1D(tbkg_val[:,"dnn"],binedges=plotrange)
	plot(normalize(hst),label="sig train", fillcolor=:cornflowerblue, alpha=0.5, lw=0, seriestype=:stepbins, fillrange=0)
	plot!(normalize(hsv),label="sig val", mc=:cornflowerblue, lw=0, ms=4, seriestype=:scatterbins)
	plot!(normalize(hbt),label="bkg train", fillcolor=:red3, alpha=0.5, lw=0, seriestype=:stepbins, fillrange=0)
	plot!(normalize(hbv),label="bkg val", mc=:red3, lw=0, ms=4, seriestype=:scatterbins)
end

# ╔═╡ 18be81da-2944-4964-ba6f-b66b9045d708
# import the function from exercise 1b to get efficiencies
function fom_for_cut(sdf::DataFrame, bdf::DataFrame, var::AbstractString, cut::Real, fom::Function; op::Function = >, tf::Function=identity, nsig::Real=0, nbkg::Real=0)
	fsig = nsig > 0 ? nsig / nrow(sdf) : 1.
	fbkg = nbkg > 0 ? nbkg / nrow(bdf) : 1.
	s = fsig*count(x -> op(tf(x), cut), sdf[!, var])
    b = fbkg*count(x -> op(tf(x), cut), bdf[!, var])
	return fom(s, b)
end

# ╔═╡ 75d4cd17-e178-4f57-b31b-94326fc962f1
let	
	cut_values = range(.0, 1.0, length=101)	
	sig_effs_dnn = Float64[]
	bkg_effs_dnn = Float64[]
	
	# Loop through cut values
	for cut in cut_values
		push!(sig_effs_dnn, fom_for_cut(tsig_val,tbkg_val,"dnn",cut,(s,b)->s,nsig=1, nbkg=1))
		push!(bkg_effs_dnn, fom_for_cut(tsig_val,tbkg_val,"dnn",cut,(s,b)->b,nsig=1, nbkg=1))		
	end
	plot(sig_effs_dnn, 1 .-bkg_effs_dnn, label="ROC DNN", lw=2, lc=2, xlabel="ε(signal)", ylabel="1-ε(background)", legend=:bottomleft, xlims = (0, 1.01), ylims = (0, 1.01))
	#deltas = diff(sig_effs_dnn)
    #cs = sig_effs_dnn[1:end-1]
	#sum(1 .-bkg_effs_dnn*Δi for (ci, Δi) in zip(cs, deltas))
end

# ╔═╡ 27216c04-292c-4049-aa41-8c18471555b7
md"""
## Exercise 2: Bias-Variance tradeoff

In this exercise, we use bootstrapped training data to estimate bias and variance of the NN training.

We first reduce the training sample size to show effects of overtraining, and to be able to train NNs on the bootstrapped datasets faster.

Your first task for this exercise: Try to build a NN that overtrains.
"""

# ╔═╡ 0e8ac9e8-fa70-4cdd-81a9-558b451ae542
begin
	# prepare data and reduce sample size
	n_training_samples = 100
	fl = [feature_names...,"label"]
	bt_train = Matrix{Float32}(vcat(tsig_train[1:n_training_samples,fl],tbkg_train[1:n_training_samples,fl]))
	n_samples = size(bt_train)[1]
	nrf = size(feature_names)[1]

	bx_train = Base.permutedims(bt_train[:,1:nrf])
	by_train = Base.permutedims(bt_train[:,end])

	bt_val = Matrix{Float32}(vcat(tsig_val[:,fl],tbkg_val[:,fl]))
	bx_val = Base.permutedims(bt_val[:,1:nrf])
	by_val = Base.permutedims(bt_val[:,end])
end

# ╔═╡ e99abe36-7fa0-4e51-ad00-3609352a06ed
begin
	Random.seed!(12345)
	b_model = Chain(Dense(nrf, 16, relu), 
					Dense(16, 16, relu),
					Dense(16, 32, relu),
					Dense(32, 16, relu),
					Dense(16, 1, sigmoid)
					)
	bopts = Flux.setup(Adam(0.03), b_model)
	bt_losses = []; bv_losses = []
end

# ╔═╡ 0a1f96a7-5c05-4adc-a130-3127f3c603e0
let
	# test the NN architecture. do we see overtraining?
	for epoch in 1:120
		t_loss, grads = Flux.withgradient(b_model) do m
			y_hat = m(bx_train)
			Flux.Losses.binarycrossentropy(y_hat, by_train)
		end
		Flux.update!(bopts, b_model, grads[1])
		v_loss = Flux.Losses.binarycrossentropy(b_model(bx_val), by_val)
		push!(bt_losses, t_loss)
		push!(bv_losses, v_loss)
	end
end

# ╔═╡ 0c4d6213-bfc7-46d3-bd69-85bbb4639762
let	
	plot(bt_losses[1:end], label="Train Loss", lw=2)
	plot!(bv_losses[1:end], label="Val Loss", lw=2, title="Loss Curve", xlabel="Epoch", ylabel="BCE")
end

# ╔═╡ 26fa1395-9ec7-4140-879f-1186eadec22e
md"""
## Exercise 2.5: Regularisation

You will now use the overtrained model and explore [regularisation](https://fluxml.ai/Flux.jl/stable/guide/training/training/#Regularisation) techniques.

The number of bootstrap samples and training epochs is preset to 100 and 120.

As a proxy to measure overtraining, we use the binary cross-entropy of the validation sample in the last epoch. Which other proxies can you think of? What would you compute if you'd be using the test sample?

One of the conceptually easiest regularisation methods are "Dropout layers" that you can add to your network architecture. Explore how the BCE plot changes with adding dropout or other regularisations.
"""

# ╔═╡ ecb4ef89-8d55-445f-9c08-b2e14ed53feb
boot_t_losses, boot_v_losses = let
	n_boot = 100
	n_epochs = 120
	# reduce the number of training events
	fl = [feature_names...,"label"]
	bt_train = Matrix{Float32}(vcat(tsig_train[:,fl], tbkg_train[:,fl]))
	nf = size(feature_names)[1]
	n_train_total = size(bt_train)[1]
	
	bx_train = Base.permutedims(bt_train[:,1:nrf])
	by_train = Base.permutedims(bt_train[:,end])
	
	bt_val = Matrix{Float32}(vcat(tsig_val[:,fl],tbkg_val[:,fl]))
	bx_val = Base.permutedims(bt_val[:,1:nf])
	by_val = Base.permutedims(bt_val[:,end])
	
	# init ouput
	t_losses = zeros(n_boot, n_epochs); v_losses = zeros(n_boot, n_epochs)
	
	for i_boot in 1:n_boot
		# get random indices with replacement for bootstrapping
		# it's very important to use a changing seed here, otherwise we'd always get the same dataset
		Random.seed!(i_boot)
		
		indices = StatsBase.sample(1:n_train_total, 2*n_training_samples, replace=true)
		t_train_boot = zeros(Float32, 2*n_training_samples, nrf+1)		
		for (i, idx) in enumerate(indices)			
			t_train_boot[i,:] = bt_train[idx,:]
		end		
		x_tb = Base.permutedims(t_train_boot[:,1:nrf])
		y_tb = Base.permutedims(t_train_boot[:,end])
		
		Random.seed!(12345)
		boot_model = Chain(Dense(nrf, 16, relu),Dropout(0.2),
					Dense(16, 16, relu),Dropout(0.2),
					Dense(16, 32, relu),Dropout(0.2),
					Dense(32, 16, relu),Dropout(0.2),
					Dense(16, 1, sigmoid)
					)
		boot_opts = Flux.setup(OptimiserChain(WeightDecay(0.42), Adam(0.03)), boot_model)
		
		for epoch in 1:n_epochs
			t_loss, grads = Flux.withgradient(boot_model) do m				
				y_hat = m(x_tb)
				Flux.Losses.binarycrossentropy(y_hat, y_tb)
			end
			Flux.update!(boot_opts, boot_model, grads[1])
			v_loss = Flux.Losses.binarycrossentropy(boot_model(bx_val), by_val)
			t_losses[i_boot,epoch] = t_loss
			v_losses[i_boot,epoch] = v_loss
		end
	end
	(t_losses, v_losses)
end

# ╔═╡ 2b313209-5327-4801-a274-778e5846b82b
let	
	# you may need to adapt the range
	plotrange = range(0.0,1,101)
	
	htl = Hist1D(boot_t_losses[:,end],binedges=plotrange)
	hvl = Hist1D(boot_v_losses[:,end],binedges=plotrange)
	plot(htl,label="training loss", fillcolor=:cornflowerblue, alpha=0.5, lw=0, seriestype=:stepbins, fillrange=0)
	plot!(hvl,label="validation loss", mc=:red3, alpha=0.5, lw=0, seriestype=:stepbins, fillrange=0, xlab="BCE", ylab="bootstrapped samples")
	
end

# ╔═╡ Cell order:
# ╠═7a20452e-6600-11f0-2e96-e195c74cdb7e
# ╠═7f6f1196-3161-4219-b3a6-6eac0fd12fe9
# ╠═c53cd813-e8be-4f97-b4d6-1f445e0df061
# ╠═3e3a86e6-d5fa-4186-93be-8ef6ab93e4b9
# ╠═b10c90b4-1acb-4a40-9b36-5c7fc80a5329
# ╠═f63e1a46-e598-4172-a330-30fdef152e9b
# ╠═99b776a0-38cc-4f34-a540-2e5e150c3bfb
# ╠═6e1d9c8a-3f01-4c8b-9568-3ff8adba497b
# ╟─f4c36064-3968-44be-8744-49e4c81415c5
# ╠═383a0230-5752-40dd-9374-e7ccc11c77ff
# ╠═933c6c00-1766-463a-ba1e-1bc8a08d65cd
# ╠═66c0d418-714e-4523-b608-4f5f2d163985
# ╠═b7d0ee07-a929-4d18-97f8-edd0dc6f0206
# ╠═28c36329-e978-485d-8d13-3079a777ce9c
# ╠═18be81da-2944-4964-ba6f-b66b9045d708
# ╠═75d4cd17-e178-4f57-b31b-94326fc962f1
# ╟─27216c04-292c-4049-aa41-8c18471555b7
# ╠═0e8ac9e8-fa70-4cdd-81a9-558b451ae542
# ╠═e99abe36-7fa0-4e51-ad00-3609352a06ed
# ╠═0a1f96a7-5c05-4adc-a130-3127f3c603e0
# ╠═0c4d6213-bfc7-46d3-bd69-85bbb4639762
# ╟─26fa1395-9ec7-4140-879f-1186eadec22e
# ╠═ecb4ef89-8d55-445f-9c08-b2e14ed53feb
# ╠═2b313209-5327-4801-a274-778e5846b82b
