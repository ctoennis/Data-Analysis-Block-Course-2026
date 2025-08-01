### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 7e4b0db4-8225-47be-b55d-f29cd37ab68b
begin
    import Pkg
    # activate a temporary environment
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="HighEnergyTools", url="https://github.com/RUB-EP1/HighEnergyTools.jl"),
		Pkg.PackageSpec(name="DataFrames"),
		Pkg.PackageSpec(name="ComponentArrays"),
		Pkg.PackageSpec(name="PlutoUI"),
		Pkg.PackageSpec(name="StatsBase"),
		Pkg.PackageSpec(name="ForwardDiff"),
		Pkg.PackageSpec(name="Zygote"),
		Pkg.PackageSpec(name="IRTools"),
		Pkg.PackageSpec(name="Parameters"),
		Pkg.PackageSpec(name="Images"),
		Pkg.PackageSpec(name="Minuit2"),
    ])
    using HighEnergyTools, Plots, HighEnergyTools.FHist, DataFrames, HighEnergyTools.Optim, ComponentArrays, PlutoUI, HighEnergyTools.QuadGK, Zygote, ForwardDiff, Parameters, IRTools, Images, StatsBase, Minuit2
end


# ╔═╡ 4c340edf-18f7-42de-aed9-9b24147ee62c
md"""
In this exercise session, we will revisit the concepts of automatic differentiation (forward and backward AD) on simple functions, draw computation graphs. Lastly, we will try fitting a compound function of a gaussian and exponential using Minuit2 and see how we can boost its performance using AD.
"""

# ╔═╡ f9c8b6df-1ddf-414e-98a0-b7ce96ddce45
md"""
### Our functions
"""

# ╔═╡ 7b881b61-b35b-42b0-abb8-dc7568b1f14d
f(x) = x^2 + sin(x) #Orignal function

# ╔═╡ 277b97f0-fd7d-4432-88d0-4abd65b5214e
#TODO Write the analytical derivate f¹(x) for our f(x) above.

# ╔═╡ 8a88d8f1-791a-419a-b32c-967fd78be3a0
F(x,y) = x^3 + sin(x)*y^2 + log(y) #Example multivariate function

# ╔═╡ 6c97ed8d-2a79-4247-8736-3b9c5c6765cd
#TODO Write the analytical derivate expressions Fˣ(x,y), Fʸ(x,y) for our F(x) above.

# ╔═╡ 427f2094-40e6-4268-8217-98899d56372b
x0 = π #We evaluate derivate/gradient at x0 for both 1D and 2D function

# ╔═╡ 10da7d82-d844-4fe7-bb6f-35de5c11378a
y0 = ℯ #We evaluate gradient at for the 2D function

# ╔═╡ b1c2c8d8-02d5-4da4-8e80-7c634f33167b
md"""
## Playing with duals and forward AD
"""

# ╔═╡ 907139b5-bf9e-4755-b8b2-70e63ca07d4a
x_dual = ForwardDiff.Dual(x0,1) # Representing a dual number (x,x'). For x, x'=1

# ╔═╡ 9f1acf28-0c72-4799-8daa-4301b428a827
f(x_dual) #f(x_dual) is also a dual (f,f')

# ╔═╡ 2777e910-baac-49f7-bbac-c94cdc450723
#TODO Write an expression to check if derivative from dual matches your analytical expression

# ╔═╡ fcd457f5-0926-4f2d-9d73-16f02eff2575
md"""
#### Lets see its action on multivariate function and see for gradients
"""

# ╔═╡ b4781012-3732-4e73-93e4-375447388aa1
y_dual = ForwardDiff.Dual(y0,1)

# ╔═╡ 3f974cb6-ebe1-450e-817d-80f4e1d4069b
md"""
Depending on the number of input duals, partial derivative directly calculated
"""

# ╔═╡ 4e8893c3-2dab-4d53-936c-7ea73fa2ca7d
F(x_dual,y0) #Returns Fˣ

# ╔═╡ 85746604-9d99-4943-bd31-968f1c38b944
F(x0,y_dual) #Returns Fʸ 

# ╔═╡ 1414890a-48e4-4908-afbf-78e3bd62722d
F(x_dual,y_dual) #What does the 'derivative' corresponds to here?

# ╔═╡ 2587b7d0-7f24-486f-acc4-623516c8435d
md"""
Write an expression for Fˣ(x,y) and Fʸ(x,y) using duals.
```
F(input_x,input_y)
```
Hint : One of the inputs will not be a dual.
"""

# ╔═╡ aa0e319b-9b3a-4bbe-b800-0f68acdcb319
#TODO Write the expression for Fˣ(x,y) here

# ╔═╡ a2acbe32-e408-4d29-8d7f-43a49c82049f
#TODO Write the expression for Fʸ(x,y) here

# ╔═╡ 6153bf11-5f77-4f2b-8f4c-826cf56b2105
md"""
### Here is what the ForwardDiff is doing to evaluate the derivative
"""

# ╔═╡ 44cb5a39-7df3-4c00-8e15-14de6d2c5550
md"""
Showing the IR (intermediate representation) steps for the evaluation of function using dual. Key things to note:
1) If a dual number is represented by (x,x'), x is accessed from dual.value and x' from dual.partials. Note that %4 and %13 are trying to access the partials for x, which is 1.
2) %1 - %8 deals with evaluation of x_dual$^2$ = (x$^2$,2x)
3) %18 == x$^2$ + sin(x) == actual value of the function
4) %19 == 2x + cos(x) == derivative of the function
5) %20-%21 corresponds to wrapping the derivative as a tuple of partials. This is used to form the final dual output.
"""

# ╔═╡ ffcf5b21-7f6d-45e0-a6a1-86b428dc338b
@code_typed f(x_dual) # f(x) = x^2 + sin(x)

# ╔═╡ cfc916bd-a53a-4675-84b3-de9e0831c4df
md"""
Same IR steps but for multivariate functions. Similar steps as above, but there are two sequential chains for each of the partials.
"""

# ╔═╡ 457796a9-a296-42be-b0ed-1a15735f8eb7
@code_typed F(x_dual,y_dual) #F(x,y) = x^3 + sin(x)*y^2 + log(y) 

# ╔═╡ 265f8f60-d2e0-426f-8ceb-44e7d90ac563
md"""
## TODO : Draw the relevant computation graphs summarizing the procedure. You can use online tools like [excalidraw](https://excalidraw.com/) for diagrams.
"""

# ╔═╡ bb564fe2-725e-4297-bc57-44fb70cdc4f0
md"""
#### Use Export Image to save your diagram and add the path to image below in 'figurespath'
"""

# ╔═╡ 91b050f6-bc25-4167-8290-3b9bc4593d32
#let
	#figurespath = joinpath("/path/to/image.png")
	#load(figurespath)
#end

# ╔═╡ 4c48d525-cc18-4470-91b8-340d3dde91cd
md"""
## Backward AD
"""

# ╔═╡ 6ec9fe36-4c15-4a6a-af79-cc97e80190d9
md"""
For backward AD, we will use Zygote package in julia. You can use Zygote.gradient to directly get the derivates and Zygote.pullback to get pullback functions as well.
"""

# ╔═╡ e3995289-c394-40d9-98cb-fddf240feaea
fx0,dfx0 = Zygote.pullback(f, x0)  # Returns the value of function at x0 and a function type object called pullback

# ╔═╡ 31c13caf-87d7-4272-9513-627924a374d3
dfx0

# ╔═╡ 53b9c618-3019-4548-81bc-b3294406484f
#TODO : Check that the derivative calculated using the pullback function matches your prediction

# ╔═╡ 3b2f4942-2684-43e5-b857-c00d32a85382
md"""
#### Checking on multivariate function
"""

# ╔═╡ 0c5e95a9-402c-4cbb-b0dc-0fa81e26dd12
Zygote.gradient(F,x0,y0)  # Returns ∂f/∂x,∂f/∂y

# ╔═╡ 915d0fb1-4442-45d2-91d4-0f1e747b18f2
Fx0y0,dFx0y0 = Zygote.pullback(F,x0,y0)

# ╔═╡ dfc9c038-d380-46d5-bd2f-0511ad377ed5
dFx0y0(1.0) #Gradients from the pullback function

# ╔═╡ b748e12f-f3a5-4f19-a0d6-bfdd761e450b
@code_ir Zygote._pullback(Zygote.Context(), f, x0) # More detailed IR output

# ╔═╡ ccef2592-0dc1-494a-9236-0d8a6a10867a
md"""
## Draw the relevant computation graphs summarizing the procedure. You can use online tools like [excalidraw](https://excalidraw.com/) for diagrams
"""

# ╔═╡ a1785dfc-951a-4c11-a11c-8dab872cd640
md"""
#### Use Export Image to save your diagram and add the path to image below in 'figurespath'
"""

# ╔═╡ 15dc58e8-e69f-4acb-ad25-0ce5e4cda681
#let
	#figurespath = joinpath("/path/to/image.png")
	#load(figurespath)
#end

# ╔═╡ 3e70912b-1936-43c8-9ab3-96bbc0b764cb
md"""
## Alright. Lets go to Minuit2 for our fitting example
"""

# ╔═╡ 7d9751b5-b2c4-47cd-af8e-40a8b996ca96
md"""
#### A plotting function defined to plot histogram as scatter points
"""

# ╔═╡ 7c8aad39-1886-4faa-9be1-3f857ea1a183
function GetHistogramScatter(data,datasupport,nbins,plottitle)
	bins = range(datasupport...; length=nbins)
	h = fit(Histogram, data, bins)

# Extract bin edges, centers, and counts
	edges = h.edges[1]
	centers = 0.5 .* (edges[1:end-1] + edges[2:end])
	counts = h.weights
	bin_width = (datasupport[2]-datasupport[1])/nbins
	#Normalize counts and errors by bin width
	#norm_counts = counts ./ bin_width
	errors = sqrt.(counts)  # Poisson errors also scaled

	#Plot histogram as data points with error bars
	return scatter(centers, counts;
    yerror = errors,
    label = "Data",title=plottitle,
    markershape = :circle,
    markersize = 4,
    legend = :topright),bins,round(bin_width,digits=4),sum(counts)
end

# ╔═╡ dfb72eaf-bd8e-4fd0-a227-81f083635331
md"""
#### A sampling function to sample datapoints using cdf for a given distribution
"""

# ╔═╡ 33fa5d99-6f7d-43bc-b26f-ee387e82e2e2
function sample_inversion(f,n,support,nbins=1000) #Function to generate samples using cdf inversion method
	MaxValue = quadgk(f,support...)[1]
	mycdf(x) = quadgk(f, support[1], x)[1]	
	grid = range(support..., nbins)
	cdf_values = mycdf.(grid)
	
	Sample=[]
	step=0
	while length(Sample)<n
		y=MaxValue*rand()
#		y=MaxValue
		binind = findfirst(cdf_values .> y) - 1
   		x_left, x_right = grid[binind], grid[binind+1]
	 
		position_inside_bin = rand()
   		x = x_left + position_inside_bin * (x_right - x_left)
    		push!(Sample,x)	
	end
	return Sample
end

# ╔═╡ c6ee7bb1-68f4-4c02-a24f-549dcc9f51b2
md"""
#### TODO : Write function signal_plus_background which takes two inputs:
x : Input variable 'x'
pars : Array of parameters (datatype Float64) 
Four fit parameters are needed:
μ : Mean of the gaussian signal function
σ : Sigma of the gaussian signal function
τ : Parameter for the exponential background function
a : Signal fraction
```
function signal_plus_background(x,pars) #Exponential + Background function
	#Extract parameters μ,σ,τ,a from array 'pars'
	signal = #Write your normalised gaussian function here.
	background = #Write your exponential function here.
	return a*signal + (1-a)*background
end
```
"""

# ╔═╡ 6b51d167-31cb-45e8-b091-04aab898a8a1
#TODO Write your function signal_plus_background here

# ╔═╡ 03570868-d903-46fd-ae42-593fe2552ba4
const gen_pars = ComponentArray(
	sig=(μ=0.7, σ=0.06),
	bgd=(τ=8,),
	a = 0.5276
)


# ╔═╡ 5505e5ab-ed23-401e-a547-ab416a4545c8
md"""
We simulate some samples for Minuit to fit. 
"""

# ╔═╡ 3e723075-1470-4515-8a86-f0f3269a555e
#DataSample = sample_inversion(x->signal_plus_background(x,collect(gen_pars)),50000,x_range)

# ╔═╡ c2863048-20da-4966-a605-73b1bce28932
md"""
Plot function to see how the distribution looks like
```
let
	SP,bins,width,total = GetHistogramScatter(DataSample,x_range,100,"Sample")
	SP
	xlabel!("x")
	ylabel!("y")
end
```
"""

# ╔═╡ 4a8da0ee-9f4c-466d-a00e-f68e9a481957
#TODO Paste the plotting code here and plot results after generating the sample

# ╔═╡ a6ed23af-af9c-41c1-a298-9cf5edfce65b
md"""
#### Function for negative log-likelihood, would be our objective to minimize.
"""

# ╔═╡ ed918078-4e88-40a4-9773-aa83c651912b
function nll(model,model_parameters,data)
	minus_sum_log = -sum(data) do x
	value = model(x, model_parameters)
    value > 0 ? log(value) : -1e10
    end
end

# ╔═╡ fe4f8e0a-d16e-490e-a535-454c1be9be90
n_sample_to_fit = 10000

# ╔═╡ 922cc272-e24a-4f80-a116-fa4d6a1278b0
objective(pars) = nll(signal_plus_background,pars,DataSample[1:n_sample_to_fit]) #Objective to be calculated for minuit to minimize

# ╔═╡ f4a2dd67-87c1-4d78-94b3-811ed04ca52c
md"""
Now we run our Minuit object and minimize it. Have a look at the Minuit summary table
```
let
	mt1=Minuit(objective,#array with initial parameters) #Defines a minuit type object
	migrad!(mt1)
	#TODO Get fit parameters and compare with your starting parameters of the function
end
```
"""

# ╔═╡ 47652b88-b610-4868-90cd-bce6afdbc0aa
#TODO Add your initial parameters in above code block to run Minuit. Ensure you know what parameter the array element corresponds to as per your fit function

# ╔═╡ a2b99d2d-9cc8-4668-83cd-9f6d8d7a0d16
md"""
We calculate the time elapsed in executing the function using @elapsed_time method. We will compare three things:
1) Time for fitting by Minuit without derivative function passed
2) Time for fitting by Minuit with ForwardDiff derivative

Please try repeating the tests by changing the number of sample points in to fit and see if it has an impact

To do that, we need a function that measures the time elapsed for a particular process.
"""

# ╔═╡ 84553114-3794-49e9-8fe4-388da2042ddb
md"""
Write your function such that it takes the cost (objective) and an array of initial parameters as inputs, and derivative as a keyword argument.
```
function ReturnOptimParsWithTime(cost,init_pars;derivative=nothing) #Discuss : What will nothing do here?
	t= @elapsed begin
		#Write your Minuit object here followed by the minimization scheme
	end
	return m,t #Returns your minimized minuit object and time elapsed for the minimization
end
```
"""

# ╔═╡ 8324513d-479c-444a-b24b-ff650bb409b4
#TODO Write your function 'ReturnOptimParsWithTime' here

# ╔═╡ f17512e3-1fa6-42f0-9a2c-6481da894f9a
md"""
Next, we need a function which computes the gradients using AD. We will use forwarddiff in our example and define a local function which calculates derivative for the objective defined above.
```
function forward_gradient_function(pars) 
	p_julia=Float64[]
	for i in pars
		push!(p_julia,i)
	end
	#TODO Complete the function so that it returns the gradient using forwarddiff
end
```
"""

# ╔═╡ 126eff1c-ac68-473a-b5b0-6e15cc2bdabc
#TODO Write your function 'forward_gradient_function' here

# ╔═╡ 73ed8dcd-4fd5-4e1e-8ce4-08332ec78986
md"""
### Check the processing times for the minimization and number of calls minuit makes for the gradient calculation
"""

# ╔═╡ 3f15183c-c928-4df5-bc1c-0087c85b8b78
#TODO Use your function ReturnOptimParsWithTime without derivative

# ╔═╡ cfed3857-5e48-439d-97c2-fd08691ada12
#TODO Use your function 'ReturnOptimParsWithTime' with derivative function 'forward_gradient_function'

# ╔═╡ 0b79b089-1ad9-4ae3-b060-57eb05bd4dfd
const x_range = (0.0, 1.5)

# ╔═╡ 242f80d8-204b-4332-8702-1193528b57b8
md"""
### Lets have a look at our fit results. Complete the code by adding parameters to your function extracted from minuit object.
"""

# ╔═╡ afd7eca1-e980-4e51-8352-015b92285efd
md"""
```
let
	SP,bins,width,total = GetHistogramScatter(DataSample,x_range,100,"Sample")
	normalization = quadgk(x_range...) do x
        signal_plus_background(x,mt1.values)
    end[1]
    dx = bins[2] - bins[1]
    n = length(DataSample)
	scale = dx * n / normalization #Scaling the function to take into account bin widths
	xlabel!("x")
	ylabel!("y")
	plot!(x->scale*signal_plus_background(x,#Minuit_without_derivative_parameters), x_range..., label="Minuit without derivative")
	plot!(x->scale*signal_plus_background(x,#Minuit_with_derivative_parameters), x_range..., label="Minuit with derivative")
end
```
"""

# ╔═╡ 9d84fdf5-87a6-49e8-91d4-985c14006a58
#TODO Write your code here

# ╔═╡ de571f59-cc1e-4603-b63c-0848db5f2ab5
md"""
Optional : Define a local function backward_gradient_function in similar way which calculates the derivative using backward diff and compare your results. Do you get similar processing times? 
"""

# ╔═╡ 3f2a81f2-34f7-40c3-9872-4a354babb7c1
md"""
#### Fit parameters. Note that passing derivatives to minuit through AD speeds up the minimization, without affecting the fit parameters
"""

# ╔═╡ Cell order:
# ╠═7e4b0db4-8225-47be-b55d-f29cd37ab68b
# ╠═4c340edf-18f7-42de-aed9-9b24147ee62c
# ╠═f9c8b6df-1ddf-414e-98a0-b7ce96ddce45
# ╠═7b881b61-b35b-42b0-abb8-dc7568b1f14d
# ╠═277b97f0-fd7d-4432-88d0-4abd65b5214e
# ╠═8a88d8f1-791a-419a-b32c-967fd78be3a0
# ╠═6c97ed8d-2a79-4247-8736-3b9c5c6765cd
# ╠═427f2094-40e6-4268-8217-98899d56372b
# ╠═10da7d82-d844-4fe7-bb6f-35de5c11378a
# ╠═b1c2c8d8-02d5-4da4-8e80-7c634f33167b
# ╠═907139b5-bf9e-4755-b8b2-70e63ca07d4a
# ╠═9f1acf28-0c72-4799-8daa-4301b428a827
# ╠═2777e910-baac-49f7-bbac-c94cdc450723
# ╠═fcd457f5-0926-4f2d-9d73-16f02eff2575
# ╠═b4781012-3732-4e73-93e4-375447388aa1
# ╟─3f974cb6-ebe1-450e-817d-80f4e1d4069b
# ╠═4e8893c3-2dab-4d53-936c-7ea73fa2ca7d
# ╠═85746604-9d99-4943-bd31-968f1c38b944
# ╠═1414890a-48e4-4908-afbf-78e3bd62722d
# ╠═2587b7d0-7f24-486f-acc4-623516c8435d
# ╠═aa0e319b-9b3a-4bbe-b800-0f68acdcb319
# ╠═a2acbe32-e408-4d29-8d7f-43a49c82049f
# ╠═6153bf11-5f77-4f2b-8f4c-826cf56b2105
# ╠═44cb5a39-7df3-4c00-8e15-14de6d2c5550
# ╠═ffcf5b21-7f6d-45e0-a6a1-86b428dc338b
# ╠═cfc916bd-a53a-4675-84b3-de9e0831c4df
# ╠═457796a9-a296-42be-b0ed-1a15735f8eb7
# ╠═265f8f60-d2e0-426f-8ceb-44e7d90ac563
# ╠═bb564fe2-725e-4297-bc57-44fb70cdc4f0
# ╠═91b050f6-bc25-4167-8290-3b9bc4593d32
# ╟─4c48d525-cc18-4470-91b8-340d3dde91cd
# ╠═6ec9fe36-4c15-4a6a-af79-cc97e80190d9
# ╠═e3995289-c394-40d9-98cb-fddf240feaea
# ╠═31c13caf-87d7-4272-9513-627924a374d3
# ╠═53b9c618-3019-4548-81bc-b3294406484f
# ╠═3b2f4942-2684-43e5-b857-c00d32a85382
# ╠═0c5e95a9-402c-4cbb-b0dc-0fa81e26dd12
# ╠═915d0fb1-4442-45d2-91d4-0f1e747b18f2
# ╠═dfc9c038-d380-46d5-bd2f-0511ad377ed5
# ╠═b748e12f-f3a5-4f19-a0d6-bfdd761e450b
# ╠═ccef2592-0dc1-494a-9236-0d8a6a10867a
# ╠═a1785dfc-951a-4c11-a11c-8dab872cd640
# ╠═15dc58e8-e69f-4acb-ad25-0ce5e4cda681
# ╠═3e70912b-1936-43c8-9ab3-96bbc0b764cb
# ╟─7d9751b5-b2c4-47cd-af8e-40a8b996ca96
# ╠═7c8aad39-1886-4faa-9be1-3f857ea1a183
# ╟─dfb72eaf-bd8e-4fd0-a227-81f083635331
# ╠═33fa5d99-6f7d-43bc-b26f-ee387e82e2e2
# ╠═c6ee7bb1-68f4-4c02-a24f-549dcc9f51b2
# ╠═6b51d167-31cb-45e8-b091-04aab898a8a1
# ╠═03570868-d903-46fd-ae42-593fe2552ba4
# ╠═5505e5ab-ed23-401e-a547-ab416a4545c8
# ╠═3e723075-1470-4515-8a86-f0f3269a555e
# ╠═c2863048-20da-4966-a605-73b1bce28932
# ╠═4a8da0ee-9f4c-466d-a00e-f68e9a481957
# ╟─a6ed23af-af9c-41c1-a298-9cf5edfce65b
# ╠═ed918078-4e88-40a4-9773-aa83c651912b
# ╠═fe4f8e0a-d16e-490e-a535-454c1be9be90
# ╠═922cc272-e24a-4f80-a116-fa4d6a1278b0
# ╠═f4a2dd67-87c1-4d78-94b3-811ed04ca52c
# ╠═47652b88-b610-4868-90cd-bce6afdbc0aa
# ╠═a2b99d2d-9cc8-4668-83cd-9f6d8d7a0d16
# ╠═84553114-3794-49e9-8fe4-388da2042ddb
# ╠═8324513d-479c-444a-b24b-ff650bb409b4
# ╠═f17512e3-1fa6-42f0-9a2c-6481da894f9a
# ╠═126eff1c-ac68-473a-b5b0-6e15cc2bdabc
# ╠═73ed8dcd-4fd5-4e1e-8ce4-08332ec78986
# ╠═3f15183c-c928-4df5-bc1c-0087c85b8b78
# ╠═cfed3857-5e48-439d-97c2-fd08691ada12
# ╠═0b79b089-1ad9-4ae3-b060-57eb05bd4dfd
# ╟─242f80d8-204b-4332-8702-1193528b57b8
# ╠═afd7eca1-e980-4e51-8352-015b92285efd
# ╠═9d84fdf5-87a6-49e8-91d4-985c14006a58
# ╠═de571f59-cc1e-4603-b63c-0848db5f2ab5
# ╟─3f2a81f2-34f7-40c3-9872-4a354babb7c1
