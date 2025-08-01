### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ d24ad8e2-5e4f-11f0-0877-c3c4393b79ac
begin
    import Pkg
    # activate a temporary environment
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="HighEnergyTools", url="https://github.com/RUB-EP1/HighEnergyTools.jl"),
		Pkg.PackageSpec(name="UnROOT"),
		Pkg.PackageSpec(name="DataFrames"),
		Pkg.PackageSpec(name="PlutoUI"),
		Pkg.PackageSpec(name="DecisionTree"),
		Pkg.PackageSpec(name="XGBoost"),
		Pkg.PackageSpec(name="TreeRecipe"),
    ])
    using HighEnergyTools, UnROOT, Plots, HighEnergyTools.FHist, DataFrames, PlutoUI, DecisionTree, XGBoost, TreeRecipe, Random, HighEnergyTools.Statistics
end

# ╔═╡ a030808c-1452-4671-9ca3-36238c91d41d
md"""
# Boosting techniques for decision trees
In this exercise we're using the same datasets as in Exercise 1B, and we're going to explore the full dataset to see if we can get a better discrimination of signal and background.
"""

# ╔═╡ 13819541-38b8-4b61-8ec5-a8f82d99e456
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
	# sanity check...
	if feature_names != names(dfbkg)
		throw("not good...")
	end
end

# ╔═╡ cc1b106c-34e6-44eb-ad72-236d95ed77b9
md"""

## Pre-processing of data

Before training a ML algorithm, it makes sense to take a closer look at the features we'd like to use for training.
"""

# ╔═╡ 3e88a4da-9727-4b18-a9ea-9f08327c21c9
md"""
A first step is to make sure that there are no values in the sample that the algorithm can potentially not handle. So, we remove NaN and other strange values from the dataframes
"""

# ╔═╡ 0fcf5805-9bac-4071-aa27-339fafa3ff3d
begin
	# apply the filter to all variables
	dfbkgf = filter(row -> all(x -> !(x isa Number && isnan(x)), row), dfbkg)
	dfsigf = filter(row -> all(x -> !(x isa Number && isnan(x)), row), dfsig)
	dftestf = filter(row -> all(x -> !(x isa Number && isnan(x)), row), dftest)
end

# ╔═╡ cb5df226-b6ad-4601-8086-7a7979e31471
md"""
Next, let's check the extrema of the data. 
"""

# ╔═╡ 96a5ca66-d6e4-46cb-b840-755a9ad07450
for fn in feature_names
	println("$fn bkg $(extrema(dfbkgf[!,fn]))")
	println("$fn sig $(extrema(dfsigf[!,fn]))")
	println("$fn test $(extrema(dftestf[!,fn]))")
	println("*"^78)
end

# ╔═╡ afdf4888-b011-4fe7-bb56-71b26456c8a1
md"""
Not all variables are in the same domain. 

In the case of the `pi_OWNPVIPCHI2` variable, there is a region, where the algorithm would see only signal (between ~1 and ~4). 

For instance, when training BDTs with AdaBoost, this variable would maximise the Gini index below values of 4, since there is only signal.

In the case of the samples used in this exercise, there were different pre-processing workflows, and the variable domains are not the same for all data.
"""

# ╔═╡ 23213d2a-7359-444b-8c90-9ef9883e22cb
md"""
We are now skipping an intermediate step in which we would plot the distributions and decide on transformations of variables.

In our case, there are a few variables where it's easier to look at the distribution with a log transform:
"""

# ╔═╡ 60d58dc8-ee29-4275-91b3-c5d8f73c58ce
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

# ╔═╡ 703a59a3-1fb9-4e0c-a49b-0cf21ab7323e
md"""
By skipping the intermediate step, the transformation and trimming of variables can be done in one go.

We can also do the splitting of signal and background samples into training and validation datasets in the same step.
"""

# ╔═╡ 1a753fe8-402a-4540-8a4d-4c8d4e7f832e
# document this function
function split_df(df, frac_train=0.8)
    n = nrow(df)
	Random.seed!(1234)
    idx = randperm(n)
    n_train = round(Int, frac_train * n)
    train_idx = idx[1:n_train]
    val_idx = idx[n_train+1:end]
    return df[train_idx, :], df[val_idx, :]
end

# ╔═╡ 5dbbf657-f4c9-4d9d-9472-82c0dce0ecbe
md"""
## Exercise 1
Let's start with something boring but very common:
Implement the transformations (in our case only log-transoformations) and trim the signal and background samples to a common support.
"""

# ╔═╡ 0544d753-2786-42b1-8d6c-fe26678b2b65
tsig_train, tsig_val, tbkg_train, tbkg_val, t_test = let	
    local_sig = copy(dfsigf)
	local_bkg = copy(dfbkgf)
	local_test = copy(dftest)
	# TODO: transformations
    for var in feature_names
		if var in log_transform
			transform!(local_sig, Symbol(var) => ByRow(log10) => Symbol(var))
			transform!(local_bkg, Symbol(var) => ByRow(log10) => Symbol(var))
			transform!(local_test, Symbol(var) => ByRow(log10) => Symbol(var))
		end
		# TODO: trim data to common support
        support_min = max(minimum(local_sig[!,var]),minimum(local_bkg[!,var]))
		support_max = min(maximum(local_sig[!,var]),maximum(local_bkg[!,var]))
		local_sig = filter(row -> row[var] > support_min && row[var] < support_max, local_sig)
		local_bkg = filter(row -> row[var] > support_min && row[var] < support_max, local_bkg)
    end
	# add labels for training and validation data
	local_sig.label .= 1  # or "signal"
    local_bkg.label .= 0  # or "background"
	# split into validation and training samples
	local_dfsigf_train, local_dfsigf_val = split_df(local_sig, 0.8)
    local_dfbkgf_train, local_dfbkgf_val = split_df(local_bkg, 0.8)
    (local_dfsigf_train, local_dfsigf_val, local_dfbkgf_train, local_dfbkgf_val, local_test)
end

# ╔═╡ f73614e5-fcf9-45da-be0c-7a8fcb43a7da
md"""
Let's look at the signal and background distributions of our feature variables. Which variables will be the most important ones to separate signal from background?
"""

# ╔═╡ cc3c2847-98ac-44fe-96c4-e31ddd29160f
let
	theme(:default ;fontfamily = "Times", grid = false, linewidth = 2, guidefontsize = 12, legendfontsize = 12, tickfontsize = 12, foreground_color_legend = :transparent)
    PP = []
	n = length(feature_names)
	for (i,var) in enumerate(feature_names)
		# for defining histograms in same range:
		plotrange = [min(minimum(tbkg_train[!,var]), minimum(tsig_train[!,var])), max(maximum(tbkg_train[!,var]), maximum(tsig_train[!,var]))]		
		# histograms for signal and background training samples
		hbkg = Hist1D(tbkg_train[!, var],binedges=range(plotrange...,100))
		hsig = Hist1D(tsig_train[!, var],binedges=range(plotrange...,100))
		# background plot
		P = plot(normalize(hbkg), label="background", fillcolor=:red3, alpha=0.5, lw=0, seriestype=:stepbins, fillrange=0)
		# signal plot
		plot!(P, normalize(hsig), label="signal", fillcolor=:cornflowerblue, alpha=0.5, title="", xlabel=(var in log_transform ? "log10($var)" : "$var"), ylabel="Candidates (norm)", guidefonthalign =:right, guidefontvalign =:top, seriestype=:stepbins, fillrange=0, lw=0, bottom_margin=5Plots.mm)
		push!(PP, P)
	end
	plot(PP...; size = default(:size) .* (1, n), layout = (n, 1))
end

# ╔═╡ e10b0edd-5818-4e6d-80b6-feea43fb6471
md"""
## Exercise 2: Training BDTs with AdaBoost
We are ready to train a BDT now.

We start with a wrapper function that helps with using dataframes, and gives us flexibility to choose the features and the number of trees to train.

Implement the function to train an `AdaBoostStumpClassifier`. You can find rudimentary documentation in the Live Docs and https://docs.juliahub.com/DecisionTree/pEDeB/0.12.4/
"""

# ╔═╡ 0e83604b-467a-419a-bc40-e49b102f9895
"""
train_ada_bdt(df::DataFrame, features::Vector{AbstractString}; n_estimators::Int)

Trains a AdaBoostStumpClassifier.

# Arguments
- `df::DataFrame`: DataFrame containing the labeled training sample.
- `features::Vector{AbstractString}`: Names of the variables (columns) to train on.
- `n_estimators::Int`: Number of shallow trees (depth=1) to grow (default=200).

# Returns
- `AdaBoostStumpClassifier`: The trained BDT.
"""
function train_ada_bdt(df, features; n_estimators=200)
	# TODO: transform the data such that it can be used to train a AdaBoostStumpClassifier, and return the trained BDT.
	X = Matrix(df[:,features])
	println(size(X))
	y = Vector(df.label)
	ada_bdt = AdaBoostStumpClassifier(n_iterations=n_estimators)
	return fit!(ada_bdt,X,y)
end

# ╔═╡ 9618a910-453c-4524-b4ba-18f35d5b06fd
md"""
To use the function, we need to prepare the labels for the training data, and merge the samples.
"""

# ╔═╡ 60f8095f-93df-496b-80bf-b8d6bc48b505
adabdt_tf = let
	tdf_train = vcat(tsig_train, tbkg_train)
	(train_ada_bdt(tdf_train, feature_names))
end

# ╔═╡ 6dc9729d-f8f8-40c1-8bfe-cac6ff630061
md"""
How does the response of the BDT look like when evaluating it on training and validation samples?
"""

# ╔═╡ 07a1b9ee-1c06-48f7-9b7d-08c3dfcca5b2
let 
	df_val = vcat(tsig_val, tbkg_val)
	
	X_sig_train = Matrix(tsig_train[:, feature_names])
	X_sig_val = Matrix(tsig_val[:, feature_names])
	X_bkg_train = Matrix(tbkg_train[:, feature_names])
	X_bkg_val = Matrix(tbkg_val[:, feature_names])
	
	response_sig_train = predict_proba(adabdt_tf, X_sig_train)[:, 2]
	response_sig_val = predict_proba(adabdt_tf, X_sig_val)[:, 2]
	response_bkg_train = predict_proba(adabdt_tf, X_bkg_train)[:, 2]
	response_bkg_val = predict_proba(adabdt_tf, X_bkg_val)[:, 2]
	tsig_val.ada_bdt = response_sig_val
	tbkg_val.ada_bdt = response_bkg_val

	plotrange = range(0.25,0.65,80)
	hst = Hist1D(response_sig_train,binedges=plotrange)
	hsv = Hist1D(response_sig_val,binedges=plotrange)
	hbt = Hist1D(response_bkg_train,binedges=plotrange)
	hbv = Hist1D(response_bkg_val,binedges=plotrange)
	plot(normalize(hst),label="sig train", fillcolor=:cornflowerblue, alpha=0.5, lw=0, seriestype=:stepbins, fillrange=0)
	plot!(normalize(hsv),label="sig val", mc=:cornflowerblue, lw=0, ms=4, seriestype=:scatterbins)
	plot!(normalize(hbt),label="bkg train", fillcolor=:red3, alpha=0.5, lw=0, seriestype=:stepbins, fillrange=0)
	plot!(normalize(hbv),label="bkg val", mc=:red3, lw=0, ms=4, seriestype=:scatterbins)
end

# ╔═╡ 632a55d5-288c-4911-9b78-23b1cd1d4ee4
md"""
What about the test sample? For this, we don't have signal and background labels. However, we have a variable which allows us to estimate the contributions from signal and background: "Lc_M"
"""

# ╔═╡ 19ce60d6-44f4-49e7-bfe7-7d8e01017ae3
begin
	y_hat_test = predict_proba(adabdt_tf, Matrix(t_test[:, feature_names]))
	println(y_hat_test[1,:])# actually y_hat returns a score/probability for both labels. They satisfy unitarity (prob(bkg)+prob(sig)=1)	
    t_test.ada_bdt = y_hat_test[:,2]
end

# ╔═╡ 9bedaa30-6ff2-41b2-9c43-cd4438c07d15
plot(t_test[!,"ada_bdt"],leg=false,seriestype=:stephist,xlab="BDT response",ylab="Candidates")

# ╔═╡ 2381634e-62e3-4d5b-91cb-41efe2a06f24
let	
	support = (2240,2340)
	binning = range(support..., length=51)
	h = Hist1D(t_test[!,"Lc_M"]; binedges=binning)
	plot(h, seriestype=:stepbins, lab="Total")
	bkg_data = filter(row -> row["ada_bdt"] < 0.45, t_test)
	h1 = Hist1D(bkg_data[!,"Lc_M"]; binedges=binning)
	plot!(h1, seriestype=:stepbins, lab="likely background")
	sig_data = filter(row -> row["ada_bdt"] > 0.45, t_test)
	h2 = Hist1D(sig_data[!,"Lc_M"]; binedges=binning)
	plot!(h2, seriestype=:stepbins, lab="likely signal")
end

# ╔═╡ c8424d61-315d-41e9-9ee7-2f0b443df838
md"""
## Exercise 3: A closer look at the results
We have trained and applied the BDT. 
Since it may still feel like a black box, let's look at the first three trees in detail, and compute the output ourselves.

Do you get the same result if you only train 3 trees, and compare the result of `predict_proba` to the manual 3rd iteration?
"""

# ╔═╡ fdf2c5c5-d3b7-40ba-9319-5ce7e05b9b27
function calculate_response(bdt, df, features; n_trees=50)
	# get the individual trees from the trained ensemble of trees
	stumps = bdt.ensemble.trees
	weights = bdt.coeffs	
	x = Matrix(df[:, features])
	n_samples = size(x, 1)
	scores = zeros(n_samples)	
	sum_w = 0
	# TODO: calculate the BDT response for every tree by hand and store them in a dataframe
	for t in 1:Int(n_trees)	    
		sum_w += weights[t]		
		predictions = DecisionTree.apply_tree(stumps[t],x)		
		scores .+= weights[t] .* predictions
		df[!,"ada_bdt_tree_$t"] = scores ./ sum_w
	end
end

# ╔═╡ a5c790a2-55b6-4cbe-9418-7c746216c781
let
	calculate_response(adabdt_tf, t_test, feature_names)
	calculate_response(adabdt_tf, tsig_val, feature_names)
	calculate_response(adabdt_tf, tbkg_val, feature_names)
end

# ╔═╡ 13bcb033-d36c-4b40-8f00-6eb2044fc2c6
plot(t_test[!,"ada_bdt_tree_3"],leg=false,seriestype=:stephist)

# ╔═╡ 495f6ca7-c9b6-4c1b-ba60-e3f777017a72
let	
	support = (2240,2340)
	binning = range(support..., length=51)
	h = Hist1D(t_test[!,"Lc_M"]; binedges=binning)
	plot(h, seriestype=:stepbins, leg=false)
	data = filter(row -> row["ada_bdt_tree_3"] < 0.15, t_test)
	h1 = Hist1D(data[!,"Lc_M"]; binedges=binning)
	plot!(h1, seriestype=:stepbins)
end

# ╔═╡ 13d5c1f8-1c6a-43da-84fd-0cb863b846be
md"""
## Visualisation of trees
To learn more about your dataset, it can be instructive to look at the trees.
"""

# ╔═╡ ccd8b5f6-5700-469f-bddc-c95fc996d3f5
for (i,name) in enumerate(feature_names)
	println("$i $name")
end

# ╔═╡ 86e9c42b-045b-4e03-9a5b-763cddd83e73
for (i, stump) in enumerate(adabdt_tf.ensemble.trees)
    println("Stump $i:")
    print_tree(stump)
end

# ╔═╡ c69c1420-59f0-4862-a79c-19ea8661a100
let
	dtree = adabdt_tf.ensemble.trees[1]
	wt = DecisionTree.wrap(dtree, (featurenames = feature_names,))	
	# plot the decision tree (implicitly calling the `TreeRecipe` plot recipe)
	# `width` and `height` of the node rectangles as well as the `size` of the 
	# plotting area are adapted in order to get a visually pleasing output
	plot(wt, .95, .95; connect_labels = ["yes", "no"])	
end

# ╔═╡ 20e129c0-026e-4edd-84c6-3d9b1f50fcfa
# copied from exercise 2
function fom_for_cut(sdf::DataFrame, bdf::DataFrame, var::AbstractString, cut::Real, fom::Function; op::Function = >, tf::Function=identity, nsig::Real=0, nbkg::Real=0)
	fsig = nsig > 0 ? nsig / nrow(sdf) : 1.
	fbkg = nbkg > 0 ? nbkg / nrow(bdf) : 1.
	s = fsig*count(x -> op(tf(x), cut), sdf[!, var])
    b = fbkg*count(x -> op(tf(x), cut), bdf[!, var])
	return fom(s, b)
end

# ╔═╡ 1a82561b-dca9-4c3c-a693-2243d22c7fd9
# comparison of full AdaBDT to the 10th iteration
let	
	i = 7
	cut_values = range(0.0, 1.0, length=100)
	sig_effs = Float64[]
	bkg_effs = Float64[]
	sig_effs_i = Float64[]
	bkg_effs_i = Float64[]
	
	# Loop through cut values
	for cut in cut_values
		push!(sig_effs, fom_for_cut(tsig_val,tbkg_val,"ada_bdt",cut,(s,b)->s,nsig=1, nbkg=1))
		push!(bkg_effs, fom_for_cut(tsig_val,tbkg_val,"ada_bdt",cut,(s,b)->b,nsig=1, nbkg=1))
		push!(sig_effs_i, fom_for_cut(tsig_val,tbkg_val,"ada_bdt_tree_$i",cut,(s,b)->s,nsig=1, nbkg=1))
		push!(bkg_effs_i, fom_for_cut(tbkg_val,tbkg_val,"ada_bdt_tree_$i",cut,(s,b)->b,nsig=1, nbkg=1))
	end
	plot(sig_effs, 1 .-bkg_effs, label="ROC full BDT", lw=2, lc=2, xlabel="ε(signal)", ylabel="1-ε(background)", legend=:bottomleft, xlims = (0, 1.01), ylims = (0, 1.01))
	plot!(sig_effs_i, 1 .-bkg_effs_i, label="ROC tree $i", lw=2, lc=3)
end

# ╔═╡ 02e7d934-424f-4b72-9017-4fd336cf4021
@bind n_gbtrees Slider(range(0,4000,4001),show_value=true,default=200)

# ╔═╡ c32a32cd-5271-457d-8619-52b49100403d
md"""
## Exercise 4: GradientBDTs
Training a gradient BDT is similar to the AdaBoost example. However, you have learned in the lecture that the boosting works in a different way. We will explore some of the most important differences together, to get an idea why GBDTs are more popular and performant for most classification tasks.
"""

# ╔═╡ 7ea44f3b-fcfe-427e-8c06-801231b404e6
gbdt, train_logloss, val_logloss = let
	tdf_train = vcat(tsig_train, tbkg_train)
	tdf_val = vcat(tsig_val, tbkg_val)
	dtrain = DMatrix(Matrix(tdf_train[:,feature_names]),label=Vector(tdf_train.label))
	dval = DMatrix(Matrix(tdf_val[:, feature_names]),label=Vector(tdf_val.label))
	watchlist = Dict(["train" => dtrain , "valid" => dval])
	params = Dict(
		:objective => "binary:logistic",  # or "reg:squarederror", "multi:softprob" etc.
		:eval_metric => "logloss",        # or "error", "auc", "rmse" etc.
		:max_depth => 2,
		:eta => 0.1,                      # learning rate
		:lambda => 1.0,                   # L2 regularization
		:alpha => 1.0,                    # L1 regularization
		:verbosity => 1,
		:watchlist => watchlist,
    )
	# we could just return the trained BDT here:
	# (xgboost(dtrain; params...))
	# for educational purposes, we do the training explicitly to monitor the loss function
	booster = Booster(dtrain; params...) # the booster will also print the losses. good to see that we get the same numbers :)
	
	# Arrays to store logloss
	train_logloss = Float64[]
	valid_logloss = Float64[]
	
	# Define logloss manually
	function logloss(y, yhat)
	    eps = 1e-15
	    yhat = clamp.(yhat, eps, 1 - eps)
	    -mean(y .* log.(yhat) .+ (1 .- y) .* log.(1 .- yhat))
	end
	
	# Manual training loop
	for i in 1:n_gbtrees
	    update!(booster, dtrain; num_round=1)
	
	    pred_train = XGBoost.predict(booster, dtrain)
	    pred_val   = XGBoost.predict(booster, dval)
	
	    push!(train_logloss, logloss(Vector(tdf_train.label), pred_train))
	    push!(valid_logloss, logloss(Vector(tdf_val.label), pred_val))
	
	    println("[$i]  train-logloss: $(train_logloss[end])  valid-logloss: $(valid_logloss[end])"), train_logloss
	end
	(booster,train_logloss,valid_logloss)
end

# ╔═╡ bc88be2e-6543-43d0-ad78-cc938949d1c9
let
	# Optional: plot the loss
	plot(1:n_gbtrees, train_logloss, label="Training LogLoss", xlabel="Boosting Round", ylabel="LogLoss", lw=2)
	plot!(1:n_gbtrees, val_logloss, label="Validation LogLoss", lw=2)
end

# ╔═╡ 41a60764-cda2-4e11-aa21-deda1e6e4023
let
	tsig_train.gbdt = XGBoost.predict(gbdt, Matrix(tsig_train[:,feature_names]))
	tbkg_train.gbdt = XGBoost.predict(gbdt, Matrix(tbkg_train[:,feature_names]))
	tsig_val.gbdt = XGBoost.predict(gbdt, Matrix(tsig_val[:,feature_names]))
	tbkg_val.gbdt = XGBoost.predict(gbdt, Matrix(tbkg_val[:,feature_names]))
end

# ╔═╡ a18ccb83-568a-439d-9f46-09e83c1057bf
let
	plotrange = range(0,1,51)
	hst = Hist1D(tsig_train[:,"gbdt"],binedges=plotrange)
	hsv = Hist1D(tsig_val[:,"gbdt"],binedges=plotrange)
	hbt = Hist1D(tbkg_train[:,"gbdt"],binedges=plotrange)
	hbv = Hist1D(tbkg_val[:,"gbdt"],binedges=plotrange)
	plot(normalize(hst),label="sig train", fillcolor=:cornflowerblue, alpha=0.5, lw=0, seriestype=:stepbins, fillrange=0)
	plot!(normalize(hsv),label="sig val", mc=:cornflowerblue, lw=0, ms=4, seriestype=:scatterbins)
	plot!(normalize(hbt),label="bkg train", fillcolor=:red3, alpha=0.5, lw=0, seriestype=:stepbins, fillrange=0)
	plot!(normalize(hbv),label="bkg val", mc=:red3, lw=0, ms=4, seriestype=:scatterbins)
end

# ╔═╡ fe167506-2dfe-4298-ba44-656a6c0e4abc
let	
	cut_values = range(0.0, 1.0, length=501)
	sig_effs_ada = Float64[]
	bkg_effs_ada = Float64[]
	sig_effs_gb = Float64[]
	bkg_effs_gb = Float64[]
	
	# Loop through cut values
	for cut in cut_values
		push!(sig_effs_ada, fom_for_cut(tsig_val,tbkg_val,"ada_bdt",cut,(s,b)->s,nsig=1, nbkg=1))
		push!(bkg_effs_ada, fom_for_cut(tsig_val,tbkg_val,"ada_bdt",cut,(s,b)->b,nsig=1, nbkg=1))
		push!(sig_effs_gb, fom_for_cut(tsig_val,tbkg_val,"gbdt",cut,(s,b)->s,nsig=1, nbkg=1))
		push!(bkg_effs_gb, fom_for_cut(tsig_val,tbkg_val,"gbdt",cut,(s,b)->b,nsig=1, nbkg=1))
	end
	plot(sig_effs_ada, 1 .-bkg_effs_ada, label="ROC AdaBDT", lw=2, lc=2, xlabel="ε(signal)", ylabel="1-ε(background)", legend=:bottomleft, xlims = (0, 1.01), ylims = (0, 1.01))
	plot!(sig_effs_gb, 1 .-bkg_effs_gb, label="ROC GBDT", lw=2, lc=3)
end

# ╔═╡ c9753920-cda7-41c2-8c41-d49ed8a3d797
md"""
The GBDT training has many more hyperparameters to tune. What happens if you play with the learning rate, the depth of trees or the regularization? 
"""

# ╔═╡ Cell order:
# ╠═d24ad8e2-5e4f-11f0-0877-c3c4393b79ac
# ╟─a030808c-1452-4671-9ca3-36238c91d41d
# ╠═13819541-38b8-4b61-8ec5-a8f82d99e456
# ╟─cc1b106c-34e6-44eb-ad72-236d95ed77b9
# ╟─3e88a4da-9727-4b18-a9ea-9f08327c21c9
# ╠═0fcf5805-9bac-4071-aa27-339fafa3ff3d
# ╟─cb5df226-b6ad-4601-8086-7a7979e31471
# ╠═96a5ca66-d6e4-46cb-b840-755a9ad07450
# ╟─afdf4888-b011-4fe7-bb56-71b26456c8a1
# ╟─23213d2a-7359-444b-8c90-9ef9883e22cb
# ╠═60d58dc8-ee29-4275-91b3-c5d8f73c58ce
# ╟─703a59a3-1fb9-4e0c-a49b-0cf21ab7323e
# ╠═1a753fe8-402a-4540-8a4d-4c8d4e7f832e
# ╟─5dbbf657-f4c9-4d9d-9472-82c0dce0ecbe
# ╠═0544d753-2786-42b1-8d6c-fe26678b2b65
# ╟─f73614e5-fcf9-45da-be0c-7a8fcb43a7da
# ╠═cc3c2847-98ac-44fe-96c4-e31ddd29160f
# ╟─e10b0edd-5818-4e6d-80b6-feea43fb6471
# ╠═0e83604b-467a-419a-bc40-e49b102f9895
# ╟─9618a910-453c-4524-b4ba-18f35d5b06fd
# ╠═60f8095f-93df-496b-80bf-b8d6bc48b505
# ╟─6dc9729d-f8f8-40c1-8bfe-cac6ff630061
# ╠═07a1b9ee-1c06-48f7-9b7d-08c3dfcca5b2
# ╟─632a55d5-288c-4911-9b78-23b1cd1d4ee4
# ╠═19ce60d6-44f4-49e7-bfe7-7d8e01017ae3
# ╠═9bedaa30-6ff2-41b2-9c43-cd4438c07d15
# ╠═2381634e-62e3-4d5b-91cb-41efe2a06f24
# ╟─c8424d61-315d-41e9-9ee7-2f0b443df838
# ╠═fdf2c5c5-d3b7-40ba-9319-5ce7e05b9b27
# ╠═a5c790a2-55b6-4cbe-9418-7c746216c781
# ╠═13bcb033-d36c-4b40-8f00-6eb2044fc2c6
# ╠═495f6ca7-c9b6-4c1b-ba60-e3f777017a72
# ╟─13d5c1f8-1c6a-43da-84fd-0cb863b846be
# ╠═ccd8b5f6-5700-469f-bddc-c95fc996d3f5
# ╠═86e9c42b-045b-4e03-9a5b-763cddd83e73
# ╠═c69c1420-59f0-4862-a79c-19ea8661a100
# ╠═20e129c0-026e-4edd-84c6-3d9b1f50fcfa
# ╠═1a82561b-dca9-4c3c-a693-2243d22c7fd9
# ╠═02e7d934-424f-4b72-9017-4fd336cf4021
# ╟─c32a32cd-5271-457d-8619-52b49100403d
# ╠═7ea44f3b-fcfe-427e-8c06-801231b404e6
# ╠═bc88be2e-6543-43d0-ad78-cc938949d1c9
# ╠═41a60764-cda2-4e11-aa21-deda1e6e4023
# ╠═a18ccb83-568a-439d-9f46-09e83c1057bf
# ╠═fe167506-2dfe-4298-ba44-656a6c0e4abc
# ╟─c9753920-cda7-41c2-8c41-d49ed8a3d797
