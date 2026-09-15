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

# ╔═╡ e96d1208-1b5f-4a3c-9554-4f2acacfa4fa
begin
    import Pkg
    # activate a temporary environment
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="HighEnergyTools", url="https://github.com/RUB-EP1/HighEnergyTools.jl"),
		Pkg.PackageSpec(name="UnROOT"),
		Pkg.PackageSpec(name="DataFrames"),
		Pkg.PackageSpec(name="ComponentArrays"),
		Pkg.PackageSpec(name="PlutoUI"),
    ])
    using HighEnergyTools, UnROOT, Plots, HighEnergyTools.FHist, DataFrames, HighEnergyTools.Optim, ComponentArrays, PlutoUI, HighEnergyTools.QuadGK
end

# ╔═╡ 1ba8acce-7b2c-4dac-acee-051a94a1bc53
begin
	# get training and validation samples
    fbkg = ROOTFile("background.root")
    fsig = ROOTFile("signal.root")
    tbkg = LazyTree(fbkg, "t", ["Lc_PT"])
	tsig = LazyTree(fsig, "t", ["Lc_PT"])
	# convert root trees to dataframes
	dfbkg = DataFrame(tbkg)
	dfsig = DataFrame(tsig)
	# remove NaN and other strange values from the dataframes
	dfbkgf = filter(row -> all(x -> !(x isa Number && isnan(x)), row), dfbkg)
	dfsigf = filter(row -> all(x -> !(x isa Number && isnan(x)), row), dfsig)
end

# ╔═╡ 78037937-d5c7-4728-9266-91f20843c1e8
begin
	# plot settings
	theme(:default ;fontfamily = "Times", grid = false, linewidth = 2, guidefontsize = 14, legendfontsize = 14, tickfontsize = 14, foreground_color_legend = :transparent)	
end

# ╔═╡ 851d9ba3-c628-446b-8892-09bf2ca9b7bc
let
	# the let .. end environment is scoped
	var = "Lc_PT"	
	# for defining histograms in same range:
	plotrange = [min(minimum(dfbkgf[!,var]), minimum(dfsigf[!,var])), max(maximum(dfbkgf[!,var]), maximum(dfsigf[!,var]))]	
	# histograms for signal and background training samples
	hbkg = Hist1D(log10.(dfbkgf[!, var]),binedges=range(log10.(plotrange)...,100))
	hsig = Hist1D(log10.(dfsigf[!, var]),binedges=range(log10.(plotrange)...,100))
	# background plot
	plot(normalize(hbkg), label="background", fillcolor=:red3, alpha=0.5, lw=0, seriestype=:stepbins, fillrange=0)
	# signal plot
	plot!(normalize(hsig), label="signal", fillcolor=:cornflowerblue, alpha=0.5, title="", xlabel="log10($var [MeV])", ylabel="Candidates (norm)", guidefonthalign =:right, guidefontvalign =:top, seriestype=:stepbins, fillrange=0, lw=0)
end

# ╔═╡ b3f2eb8b-1674-402c-b39a-46508ecf8dc2
md"""
# Exercise 1
We would like to calculate the Gini Index and $\frac{S}{\sqrt{S+B}}$ when applying a selection on the Lc\_PT variable.

   a) Discuss how you would to this for a given cut value, and the signal and background samples at hand. 
   
   b) Do you expect to get the same result when measuring the number on actual data?

   c) Write a generic function that computes any figure of merit, given a selection, signal and background samples. The function documentation is already given below.
"""

# ╔═╡ 532fa328-614e-4cd0-8c46-32a4f1e1756d
"""
fom_for_cut(sdf::DataFrame, bdf::DataFrame, var::AbstractString, cut::Real, fom::Function; tf::Function=identity)

Compute a figure of merit (FoM) for a given cut value on a specified variable.

# Arguments
- `sdf::DataFrame`: DataFrame containing the signal sample.
- `bdf::DataFrame`: DataFrame containing the background sample.
- `var::AbstractString`: Name of the variable (column) to cut on.
- `cut::Real`: Threshold value; events with `tf(row[var]) > cut` are selected.
- `fom::Function`: A function `fom(s, b)` that computes the figure of merit from the number of signal (`s`) and background (`b`) events.
- `op::Function=>(a, b)`: Comparison operator (e.g. `>`, `<`, `>=`, `<=`). The default is `>`.
- `tf::Function=identity`: An optional transformation applied to the variable before cutting (e.g., `log10`).
- `nsig::Real`: Use this optional number of signal events, instead of the number of events in sdf.
- `nbkg::Real`: Use this optional number of background events, instead of the number of events in bdf.

# Returns
- `Float64`: The computed figure of merit for the given cut.

# Example
```julia
  fom(s, b) = s / sqrt(s + b)
  fom_for_cut(signal_df, background_df, "Lc_PT", 3.5, fom; tf=log10)
```
"""
function fom_for_cut(sdf::DataFrame, bdf::DataFrame, var::AbstractString, cut::Real, fom::Function; op::Function = >, tf::Function=identity, nsig::Real=0, nbkg::Real=0)
	fsig = nsig > 0 ? nsig / nrow(sdf) : 1.
	fbkg = nbkg > 0 ? nbkg / nrow(bdf) : 1.
	s = fsig*count(x -> op(tf(x), cut), sdf[!, var])
    b = fbkg*count(x -> op(tf(x), cut), bdf[!, var])
	return fom(s, b)
end

# ╔═╡ 832dee5f-3f5b-4761-8414-b705b0b99f0e
md"""
# Exercise 2
a) Use the function to scan cut values and produce a plot of the Gini Index and $\frac{S}{\sqrt{S+B}}$. Think about a reasonable range and transformation of the variable. Hint: normalize to plot both curves in the same pad.

b) Plot the ROC curve, and add a marker for the point where the figures of merit are maximal. Hint: you will need signal and background efficiencies for the ROC curve. They can be obtained from the generic function we wrote earlier.

c) Discuss the results.
"""

# ╔═╡ e13eb3c0-245d-44a3-9389-691e737343f1
@bind nsig Slider(range(0,100000,10001),show_value=true,default=32150)

# ╔═╡ 414f87b6-843f-4efa-ace4-6e45939f6429
@bind nbkg Slider(range(0,1000000,10001),show_value=true,default=215599)

# ╔═╡ 408e7550-7341-48c8-a389-66b71f609d8b
begin  
	# split up this block to computation and plotting
	# Define the cut range
	cut_values = range(3.2, 4.0, length=120)
	
	# Initialize storage
	ginis = Float64[]
	significances = Float64[]
	sig_effs = Float64[]
	bkg_effs = Float64[]
	
	# Loop through cut values
	for cut in cut_values	    
	    push!(ginis, fom_for_cut(dfsigf,dfbkgf,"Lc_PT",cut,(s,b)->s/sqrt(s+b); tf=log10, nsig, nbkg))
	    push!(significances, fom_for_cut(dfsigf,dfbkgf,"Lc_PT",cut,(s,b)->2*s*b/(s+b)^2; tf=log10, nsig, nbkg))
		push!(sig_effs, fom_for_cut(dfsigf,dfbkgf,"Lc_PT",cut,(s,b)->s,tf=log10, nsig=1, nbkg=1))
		push!(bkg_effs, fom_for_cut(dfsigf,dfbkgf,"Lc_PT",cut,(s,b)->b,tf=log10, nsig=1, nbkg=1))
	end
	best_index_gini = argmax(ginis)
	best_index_significances = argmax(significances)
	best_cut_gini = cut_values[best_index_gini]
	best_cut_significances = cut_values[best_index_significances]
	print("Best cut gini: $best_cut_gini\n")
	print("Best cut best_cut_significances: $best_cut_significances")
end

# ╔═╡ 99a5c90d-493d-4dc1-8e08-e5771d7891ab
let
	# lot normalized ginis and significances as function of the cut values
	plot(cut_values, normalize(ginis), label="Gini", lw=2, lc=32, xlabel="cut on log10(Lc_PT)", ylabel="Metric", legend=:topleft)
	plot!(cut_values, normalize(significances), label="Significance", lw=2, linestyle=:dash, lc=651)
	vline!([best_cut_gini],label="", lc=32)
	vline!([best_cut_significances], lw=2, linestyle=:dash, label="",lc=651)
end

# ╔═╡ 091b230f-3bcb-49c4-b2f9-40972af898c6
let
	# Some print statements to see if everything works as expected
	print("Signal efficiency gini: $(sig_effs[best_index_gini])\n")
	print("Background rejection gini: $(1 - bkg_effs[best_index_gini])\n")
	print("Signal efficiency significance: $(sig_effs[best_index_significances])\n")
	print("Background rejection significance: $(1 - bkg_effs[best_index_significances])\n")
	# Plot ROC curve
	plot(sig_effs, 1 .-bkg_effs, label="ROC", lw=2, lc=2, xlabel="ε(signal)", ylabel="1-ε(background)", legend=:bottomleft, xlims = (0, 1.01), ylims = (0, 1.01))
	# Add markers at best Gini and best significance
	scatter!([sig_effs[best_index_gini]], [1 - bkg_effs[best_index_gini]], 
	    label="Best Gini", marker=:circle, color=32, ms=6)
	scatter!([sig_effs[best_index_significances]], [1 - bkg_effs[best_index_significances]], 
	    label="Best Significance", marker=:square, color=651, ms=6)	
end

# ╔═╡ b56671f2-9e34-41de-a5a0-cf5c84a2201d
md"""
# Exercise 3
We will apply the previously found best cut-values to the test dataset, and fit the signal in the variable "Lc_M" together. Then we discuss the results.
"""

# ╔═╡ 08630e25-20f0-4f5b-8817-1acc286c67d8
begin	
    ftest = ROOTFile("test.root")
    ttest = LazyTree(ftest, "t", ["Lc_PT","Lc_M"])
	dftest = DataFrame(ttest)
	# remove NaN and other strange values from the dataframes
	dftestf = filter(row -> all(x -> !(x isa Number && isnan(x)), row), dftest)
end

# ╔═╡ b9a0c18d-dc3b-4a0d-b262-d80088751b54
begin
	support = (2240,2340)
	binning = range(support..., length=51)	
	anka = Anka(support...)
    pars = ComponentArray(sig = (μ = 2287, σ = 5), bgd = (coeffs = [1.2,0.3],), logfB = 1.8)
end

# ╔═╡ 67a95ae5-e725-4a86-9b65-3c7c5a4d9bc1
let
	# let's have a look at the data and the model with the starting parameters
    h = Hist1D(dftestf[!,"Lc_M"]; binedges=binning)
	plot(h, seriestype=:stepbins)
    plot!(x->pdf(build_model(anka,pars),x), WithData(h), leg=false)
end

# ╔═╡ f94251fb-1103-4d66-905e-ef3816627764
"""
fit_data_with_anka(tdf::DataFrame, var::AbstractString, cut::Real; op::Function = >, tf::Function=identity)

Fits a dataset with a pre-defined Anka model.

# Arguments
- `log10_pt_cut::Real`: apply a cut on log10(Lc_PT) before fitting
- `tdf::DataFrame`: DataFrame containing the sample to fit.
- `var::AbstractString`: Name of the variable (column) to cut on.
- `cut::Real`: Threshold value; events with `op(tf(row[var]), cut)` are selected.
- `op::Function=>(a, b)`: Comparison operator (e.g. `>`, `<`, `>=`, `<=`). The default is `>`.
- `tf::Function=identity`: An optional transformation applied to the variable before cutting (e.g., `log10`).
# Returns
- `MixtureModel`: Model with best fit parameters

# Example
```julia
  
```
"""
function fit_data_with_anka(tdf::DataFrame, var::AbstractString, cut::Real; op::Function = >, tf::Function=identity)
	data = filter(row -> op(tf(row[var]), cut) && support[1] < row["Lc_M"] < support[2], tdf)
	fit_res = fit_nll(p->build_model(anka,p), data[!,"Lc_M"], pars)
    best_pars = fit_res.minimizer
	print(best_pars)
    return (data=data, model=build_model(anka, best_pars))
end

# ╔═╡ 56aa718c-a1d9-4b78-8ff5-4eb3aec4c5e7
function get_yields_denominator()
	fr = fit_data_with_anka(dftestf, "Lc_PT", 0, tf=log10)
	h = Hist1D(fr.data[!,"Lc_M"]; binedges=binning)
	plot(h, seriestype=:scatterbins, ms=3, mc=:black, ylims= (0, :auto), yerror=binerrors(h))
    plt = plot!(x->pdf(fr.model,x), WithData(h), legend=false, lc=:cornflowerblue)
	fit_fractions = pdf(fr.model.prior)
	nevents = nrow(fr.data)
	return (nsig=nevents*fit_fractions[1], nbkg=nevents*fit_fractions[2], plt=plt)
end

# ╔═╡ b758333c-216b-43e0-907d-d007fd5933e0
begin
	results = get_yields_denominator()	
	nsig_denominator = results.nsig
	nbkg_denominator = results.nbkg
	results.plt
end

# ╔═╡ 0b9e0549-66e4-401f-935d-fd77f414191f
let
	fr = fit_data_with_anka(dftestf, "Lc_PT", best_cut_gini, tf=log10)
	h = Hist1D(fr.data[!,"Lc_M"]; binedges=binning)    
	fit_fractions = pdf(fr.model.prior)
	nevents = nrow(fr.data)
	nsig = nevents*fit_fractions[1] 
	nbkg = nevents*fit_fractions[2]
	eff_sig = nsig / nsig_denominator
	eff_bkg = nbkg / nbkg_denominator
	print("\n")
	print("\n")
	println("Gini signal efficiency: $eff_sig")
	println("Gini background rejection: $(1-eff_bkg)")
	#plot
	plot(h, seriestype=:scatterbins, ms=3, mc=:black, ylims= (0, :auto), yerror=binerrors(h))
    plot!(x->pdf(fr.model,x), WithData(h), legend=false, lc=:cornflowerblue)
end

# ╔═╡ 60a617ef-2e77-404b-b3db-dd7ff9da0ce2
let
	fr = fit_data_with_anka(dftestf, "Lc_PT", best_cut_significances, tf=log10)
	h = Hist1D(fr.data[!,"Lc_M"]; binedges=binning)
    
	fit_fractions = pdf(fr.model.prior)
	nevents = nrow(fr.data)
	nsig = nevents*fit_fractions[1] 
	nbkg = nevents*fit_fractions[2]
	eff_sig = nsig / nsig_denominator
	eff_bkg = nbkg / nbkg_denominator
	print("\n")
	print("\n")
	println("SsqrtSB signal efficiency: $eff_sig")
	println("SsqrtSB background rejection: $(1-eff_bkg)")

	plot(h, seriestype=:scatterbins, ms=3, mc=:black, ylims= (0, :auto), yerror=binerrors(h))
    plot!(x->pdf(fr.model,x), WithData(h), legend=false, lc=:cornflowerblue)
end

# ╔═╡ Cell order:
# ╠═e96d1208-1b5f-4a3c-9554-4f2acacfa4fa
# ╠═1ba8acce-7b2c-4dac-acee-051a94a1bc53
# ╠═78037937-d5c7-4728-9266-91f20843c1e8
# ╠═851d9ba3-c628-446b-8892-09bf2ca9b7bc
# ╠═b3f2eb8b-1674-402c-b39a-46508ecf8dc2
# ╠═532fa328-614e-4cd0-8c46-32a4f1e1756d
# ╠═832dee5f-3f5b-4761-8414-b705b0b99f0e
# ╠═e13eb3c0-245d-44a3-9389-691e737343f1
# ╠═414f87b6-843f-4efa-ace4-6e45939f6429
# ╠═408e7550-7341-48c8-a389-66b71f609d8b
# ╠═99a5c90d-493d-4dc1-8e08-e5771d7891ab
# ╠═091b230f-3bcb-49c4-b2f9-40972af898c6
# ╠═b56671f2-9e34-41de-a5a0-cf5c84a2201d
# ╠═08630e25-20f0-4f5b-8817-1acc286c67d8
# ╠═b9a0c18d-dc3b-4a0d-b262-d80088751b54
# ╠═67a95ae5-e725-4a86-9b65-3c7c5a4d9bc1
# ╠═f94251fb-1103-4d66-905e-ef3816627764
# ╠═56aa718c-a1d9-4b78-8ff5-4eb3aec4c5e7
# ╠═b758333c-216b-43e0-907d-d007fd5933e0
# ╠═0b9e0549-66e4-401f-935d-fd77f414191f
# ╠═60a617ef-2e77-404b-b3db-dd7ff9da0ce2
