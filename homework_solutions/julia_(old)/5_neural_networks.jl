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

# ╔═╡ a2057ae6-869f-4dea-b388-5a885325c395
begin
    import Pkg
    # activate a temporary environment
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="HighEnergyTools", url="https://github.com/RUB-EP1/HighEnergyTools.jl"),
		Pkg.PackageSpec(name="DataFrames"),
		Pkg.PackageSpec(name="PlutoUI"),
    	Pkg.PackageSpec(name="Flux"),
	])
    using HighEnergyTools, Plots, DataFrames, PlutoUI, HighEnergyTools.QuadGK, Flux, HighEnergyTools.Statistics, HighEnergyTools.Random
end

# ╔═╡ 771d7bf5-f25f-414a-924b-e6c47cb7afd7
TableOfContents()

# ╔═╡ 215cd2be-7fe2-4e04-895f-248d82d0ef8e
md"""
# Exercise session 3 : Training neural networks
In these exercises, you will train a simple neural network using Flux.jl package of Julia. You will try to model polynomial functions and try to understand impact of activation functions (we will test relu and tanh, you can try one more activation function of your choice). If time permits, we can also discuss a bit about hyperparameter tuning.
"""

# ╔═╡ 50fda67c-5d19-47ee-a993-13951ae20076
md"""
### Defining our polynomial functions
"""

# ╔═╡ 6f832b95-03a1-498b-bd9d-a9f0f0d7bd2c
f1(x) = 0.5*x^2 + 1.2*x - 2.5 

# ╔═╡ 8b97ec1f-303c-46b6-b9d3-1768c9d46cf4
f2(x) = x^3 + 2*x^2 + 5*x + 1

# ╔═╡ ba452564-4d14-4f1f-ad9e-f2c4d86e30ff
f3(x) = x^5 + 3*x^4 - 2.8*x^3 + 4.9*x^2 - 0.5*x + 3

# ╔═╡ d4c0bde6-8ec2-4a3a-ac58-c72aa69c15d1
# f4(x) = polynomial of your choice

# ╔═╡ 2f4e21f5-d3d3-4de4-a692-a53be8cb54fb
md"""
## Preparing the sample
"""

# ╔═╡ debc7bf3-aef3-4b29-a2b4-b58fdeb8d806
md"""
**n_sample** : $(@bind n_sample Slider(0:100:5000, show_value=true,default=1000)) Number of sample points to simulate for datasets
"""

# ╔═╡ 1c0f87e9-ee3b-4a6b-a0de-ca44a35a7366
md"""
Split fractions to filter out training, validation and test datasets from the sample. You can tune train and validation fraction here, test fraction is assumed to be 1 - train fraction- validation fraction.\
**train split fraction** : $(@bind split_ratio_train Slider(0:0.1:1.0, show_value=true,default=0.4)) \
**validation split fraction** : $(@bind split_ratio_val Slider(0:0.1:1.0, show_value=true,default=0.4)) \
"""

# ╔═╡ 6a7a51de-2544-4719-828f-4d8eb7bb5a38
md"""
As done yesterday, we will create a sample of random points and try to train a neural network with it to model y = poly(x) where poly(x) is a polynomial function (you can use one of the polynomials or define your own one in the available cell). Note that for training the network, it would be ideal to normalize the y values using mean and standard deviation as: $y_{norm} = \frac{y-\mu_y}{\sigma_y}$
```
begin
	# Generate data
	x_low = -2
	x_up = 2
	xs = Float32.(x_low .+ (x_up-x_low) .* rand(n_sample))
	ys = Float32.(<polynomial_function>.(xs) .+ 0.1 .* randn(n_sample))  # Add noise
	data = DataFrame()
	data[!,"x"] = xs
	data[!,"y"] = ys
	y_mean = #Write the mean here 
	y_std = #Write the standard deviation here
	data[!,"ynorm"] = #Write the normalized yvalues here. 

	# Split into train and validation sets
	shuffle!(MersenneTwister(123),data) #Shuffle the dataframe
	n_train = Int(floor(split_ratio_train * n_sample))
	n_val = Int(floor(split_ratio_val * n_sample))
	
	#Now that the dataframe is shuffled, we select the first few rows for training, next few rows for validation and remaining for test. 
	x_train = data[...]
	y_train = data[...]
	x_val = data[...]
	y_val = data[...]
	x_test = data[...]
	y_test = data[...]
	
	# Reshape the corresponding sets in format 1xN using reshape method
	x_train_batch = reshape(...)
	y_train_batch = reshape(...)
	x_val_batch = reshape(...)
	y_val_batch = reshape(...)
	x_test_batch = reshape(...)
	y_test_batch = reshape(...)
	sorted_x_test_vector = sort(x_test) #This we will need later while making predictions :)
end
```
"""

# ╔═╡ c36237c5-c7ab-411d-b252-6850b5c5f05f
begin
	# Generate data
	x_low = -2
	x_up = 2
	xs = Float32.(x_low .+ (x_up-x_low) .* rand(n_sample))
	ys = Float32.(f1.(xs) .+ 0.01 .* randn(n_sample))  # Add noise
	data = DataFrame()
	data[!,"x"] = xs
	data[!,"y"] = ys
	y_mean = mean(ys) #Write the mean here 
	y_std = std(ys) #Write the standard deviation here
	data[!,"ynorm"] = (ys .- y_mean) ./ y_std #Write the normalized yvalues here. 

	# Split into train and validation sets
	shuffle!(MersenneTwister(123),data) #Shuffle the dataframe
	n_train = Int(floor(split_ratio_train * n_sample))
	n_val = Int(floor(split_ratio_val * n_sample))
	
	#Now that the dataframe is shuffled, we select the first few rows for training, next few rows for validation and remaining for test. 
	x_train = data[1:n_train,:x]
	y_train = data[1:n_train,:ynorm]
	x_val = data[n_train+1:n_train+n_val,:x]
	y_val = data[n_train+1:n_train+n_val,:ynorm]
	x_test = data[n_train+n_val+1:end,:x]
	y_test = data[n_train+n_val+1:end,:ynorm]
	
	# Reshape the corresponding sets in format 1xN using reshape method
	x_train_batch = reshape(x_train,1,:)
	y_train_batch = reshape(y_train,1,:)
	x_val_batch = reshape(x_val,1,:)
	y_val_batch = reshape(y_val,1,:)
	x_test_batch = reshape(x_test,1,:)
	y_test_batch = reshape(y_test,1,:)
	sorted_x_test_vector = sort(x_test) #This we will need later while making predictions :)
end

# ╔═╡ a76dbf3e-ed2f-4f50-9a8d-8c072befd1d4
md"""
Lets visualize our datasets by plotting them. DO NOT PLOT THE TEST DATASET!
What do you expect for the train and validation datasets to look like?
```
let
	plot(title="Data distribution")
	scatter!(...,label="Train Data")
	scatter!(...,label="Validation Data")
	xlabel!("x")
	ylabel!("y")
end
```
"""

# ╔═╡ 9714ca80-59c1-4006-9035-e6b8f83542d2
let
	plot(title="Data distribution")
	scatter!(x_train,y_train,label="Train Data")
	scatter!(x_val,y_val,label="Validation Data")
	xlabel!("x")
	ylabel!("y")
end

# ╔═╡ 50335286-1c36-4f06-97cc-1dc42c815bc9
md"""
## Training with relu activation function
"""

# ╔═╡ 98d1a8ad-f1d4-41ce-a111-6a0c9c42155c
md"""
**n_epochs** : $(@bind epochs Slider(0:100:5000, show_value=true,default=1000))  
Number of training steps to update the weights
"""

# ╔═╡ 6fb19698-c20f-4399-bd0e-4ee1a61488cf
md"""
**wait_period** : $(@bind wait_period Slider(0:100:epochs, show_value=true,default=100))  
Number of epochs to wait before applying early stopping
"""

# ╔═╡ bd009118-542d-423b-94fb-4eeb9d3d8394
opt = Descent(0.05) #Setting the optimizer for the model. One can also use Adam optimizer.

# ╔═╡ 8a40a81f-8bda-4492-be94-833810dc18c2
md"""
Now we come to defining the model and writing the training loop for the neural network. Some things to do:
1) Visualize the neural network using the architecture defined. You can use Excalib tool to draw it :)
2) For starters, let it train for all the epochs and plot the loss curves (for training and validation samples) in the next cell
3) Implement early stopping criteri
```
begin
	# Define neural network model here. We use dense layers and currently using relu activation function to define our neural network. 
	model = Chain(
	    Dense(1 => 8, relu),
	    Dense(8 => 8, relu),
	    Dense(8 => 1)
	)
	
	# Track losses
	train_losses = Float64[] #Loss for training dataset
	val_losses = Float64[] #Loss for validation dataset
	training_epochs = Float64[]
	opt_state = Flux.setup(opt, model) #One has to define the state of the model based on the initialized weights and optimizer before starting the training
	best_val_loss = Flux.mse(model(x_val_batch),y_val_batch) #Calculating initial validation loss, will be used for early stopping
	wait_time = 1 #loop variable for early stopping criteria
	for epoch in 1:epochs
	    # Gradient step
	    grads = Flux.gradient(model) do m
	        Flux.mse(m(x_train_batch),y_train_batch) #Gradients are calculated
	    end
		Flux.update!(opt_state, model, grads[1]) #Weights updated by the optimizer
		
		# Record losses
	    train_loss = #Use Flux.mse to evaluate this loss for training data
	    val_loss = #Use Flux.mse to evaluate this loss for validation data
	    push!(train_losses, train_loss)
	    push!(val_losses, val_loss)
		push!(training_epochs,epoch)

		# Applying early stopping (After training and inspecting the loss curves)
			#Write your code for early stopping criteria here. You compare the val_loss with the best_val_loss, if they dont differ by some small factor δ, then you increment the wait_time loop variable; else you replace best_val_loss with current val_loss. Once the wait_time reaches wait_period, we terminate training.

	    # Print every 100 epochs
	    if epoch % 100 == 0
	        println("Epoch $epoch - Train Loss: $(round(train_loss, digits=4)), Val Loss: $(round(val_loss, digits=4))")
	    end
	end
end
```
"""

# ╔═╡ e9bb0ec2-6d3f-4025-8199-0bd976d0c0ba
begin
	# Define neural network model here. We use dense layers and currently using relu activation function to define our neural network. 
	relu_model = Chain(
	    Dense(1 => 8, relu),
	    Dense(8 => 8, relu),
	    Dense(8 => 1)
	)
	
	# Track losses
	train_losses = Float64[] #Loss for training dataset
	val_losses = Float64[] #Loss for validation dataset
	training_epochs = Float64[]
	opt_state = Flux.setup(opt, relu_model) #One has to define the state of the model based on the initialized weights and optimizer before starting the training
	best_val_loss = Flux.mse(relu_model(x_val_batch),y_val_batch) #Calculating initial validation loss, will be used for early stopping
	wait_time = 1 #loop variable for early stopping criteria
	for epoch in 1:epochs
	    # Gradient step
	    grads = Flux.gradient(relu_model) do m
	        Flux.mse(m(x_train_batch),y_train_batch) #Gradients are calculated
	    end
		Flux.update!(opt_state, relu_model, grads[1]) #Weights updated by the optimizer
		
		# Record losses
	    train_loss = Flux.mse(relu_model(x_train_batch),y_train_batch) #Use Flux.mse to evaluate this loss for training data
	    val_loss = Flux.mse(relu_model(x_val_batch),y_val_batch) #Use Flux.mse to evaluate this loss for validation data
	    push!(train_losses, train_loss)
	    push!(val_losses, val_loss)
		push!(training_epochs,epoch)

		# Applying early stopping (After training and inspecting the loss curves)
			#Write your code for early stopping criteria here. You compare the val_loss with the best_val_loss, if they dont differ by some small factor δ, then you increment the wait_time loop variable; else you replace best_val_loss with current val_loss. Once the wait_time reaches wait_period, we terminate training.
		if abs(val_loss - best_val_loss) < 0.001
			wait_time += 1
		else
			best_val_loss = val_loss
		end
		if wait_time == wait_period
			println("Validation loss hasnt improved much for $(wait_period) epochs. Stopping the training at epoch $(epoch)")
			break
		end
		
	    # Print every 100 epochs
	    if epoch % 100 == 0
	        println("Epoch $epoch - Train Loss: $(round(train_loss, digits=4)), Val Loss: $(round(val_loss, digits=4))")
	    end
	end
end

# ╔═╡ 4ae9326a-ed88-4048-a85b-124d42fe8faf
md"""
Once the training is done, lets see how the losses change over training.
```
let
	plot(title="Evolution of loss")
	plot!(...,label="train loss")
	plot!(...,label="validation loss")
	xlabel!("Epoch")
	ylabel!("MSE")
end
```
"""

# ╔═╡ a0d842ee-8740-48f5-b994-9f405763d5d8
let
	plot(title="Evolution of loss")
	plot!(training_epochs,train_losses,label="train loss")
	plot!(training_epochs,val_losses,label="validation loss")
	xlabel!("Epoch")
	ylabel!("MSE")
end

# ╔═╡ c810672c-11d9-45c7-8a91-d48e9f8177e0
md"""
### The ReLU function linearizes the output and its effect is visible in the output of the model.
"""

# ╔═╡ 294e5c00-14b8-4e0f-8109-fd1c8335fbc9
md"""
Lastly, we check the performance of the model on test dataset
```
let 
	sorted_x_test_batch = reshape(...) #Reshape the sorted_x_test_vector here
	y_pred = (vec(model(sorted_x_test_batch))  
	plot(title="Testing output for ReLU model")
	scatter!(vec(x_test_batch),vec(y_test_batch),label="Data")
	plot!(...,label="Prediction")
end
```
"""

# ╔═╡ 6f716900-b826-4162-b63b-e62de3ed5d15
let 
	sorted_x_test_batch = reshape(sorted_x_test_vector,1,:) #Reshape the sorted_x_test_vector here
	y_pred = vec(relu_model(sorted_x_test_batch))  
	plot(title="Testing output for ReLU model")
	scatter!(vec(x_test_batch),vec(y_test_batch),label="Data")
	plot!(sorted_x_test_vector,y_pred,label="Prediction")
end

# ╔═╡ 8515e2da-6d31-4179-a838-2cf6f1a22ad8
md"""
## Training with tanh activation function
"""

# ╔═╡ d6f71020-1678-41cd-b508-8c4c40758349
md"""
Now we define a model with same architecture as before but with tanh activation function. Complete the training loop below, you can copy paste the missing sections from earlier training loop. 
```
begin
	tanh_model = #Define your model here
	
	# Track loss
	train_losses_tanh = Float64[]
	val_losses_tanh = Float64[]
	training_epochs_tanh = Float64[]
	new_opt_state = Flux.setup(opt, ...) #Using the same optimizer
	best_val_loss_tanh = #Use Flux.mse to define this variable
	wait_time_tanh = 1 #loop variable for early stopping criteria
	for epoch in 1:epochs
	    # Write the steps for gradient calculation and update of weights here from earlier training loop
		
		# Record losses
	    train_loss_tanh = #Use Flux.mse to evaluate this loss for training data
	    val_loss_tanh = #Use Flux.mse to evaluate this loss for validation data
	    push!(train_losses_tanh, train_loss_tanh)
	    push!(val_losses_tanh, val_loss_tanh)
		push!(training_epochs_tanh,epoch)

		#Write the code for early stopping criteria here
		
	    # Print every 100 epochs
	    if epoch % 100 == 0
			#println("Checking model weights at $(epoch): $(model[1].weight)")
	        println("Epoch $epoch - Train Loss: $(round(train_loss_tanh, digits=4)), Val Loss: $(round(val_loss_tanh, digits=4))")
	    end
	end
end
```
"""

# ╔═╡ 605ee93a-46ce-40d8-8bed-db901fe8372c
begin
	# Define neural network model
	tanh_model = Chain(
	    Dense(1 => 8, tanh),#;init=Flux.randn32(MersenneTwister(Int(floor(rand(1)[1]*10))))),
	    Dense(8 => 8, tanh),#;init=Flux.randn32(MersenneTwister(Int(floor(rand(1)[1]*10))))),
	    Dense(8 => 1)
	)
	
	# Track loss
	train_losses_tanh = Float64[]
	val_losses_tanh = Float64[]
	training_epochs_tanh = Float64[]
	new_opt_state = Flux.setup(opt, tanh_model)
	best_val_loss_tanh = Flux.mse(tanh_model(x_val_batch),y_val_batch)
	wait_time_tanh = 1
	for epoch in 1:epochs
	    # Gradient step
	    grads = Flux.gradient(tanh_model) do m
	        Flux.mse(m(x_train_batch),y_train_batch)
	    end
		Flux.update!(new_opt_state, tanh_model, grads[1])
		
		# Record losses
	    train_loss_tanh = Flux.mse(tanh_model(x_train_batch),y_train_batch)
	    val_loss_tanh = Flux.mse(tanh_model(x_val_batch),y_val_batch)
	    push!(train_losses_tanh, train_loss_tanh)
	    push!(val_losses_tanh, val_loss_tanh)
		push!(training_epochs_tanh,epoch)

		# Applying early stopping
		if abs(val_loss_tanh - best_val_loss_tanh) < 0.001
			wait_time_tanh += 1
		else
			best_val_loss_tanh = val_loss_tanh
		end
		if wait_time_tanh == wait_period
			println("Validation loss hasnt improved much for $(wait_period) epochs. Stopping the training at epoch $(epoch)")
			break
		end
	    # Print every 100 epochs
	    if epoch % 100 == 0
	        println("Epoch $epoch - Train Loss: $(round(train_loss_tanh, digits=4)), Val Loss: $(round(val_loss_tanh, digits=4))")
	    end
	end
end

# ╔═╡ 5ba69755-ba82-441b-b8a5-1b354b262b43
md"""
Plot the loss values now for tanh model
```
let
	plot(title="Evolution of loss for tanh model")
	plot!(...,label="train loss")
	plot!(...,label="validation loss")
	xlabel!("Epoch")
	ylabel!("MSE")
end
```
"""

# ╔═╡ 4ccdbf9f-b280-4677-a852-f3b3552bd9f5
let
	plot(title="Evolution of loss")
	plot!(training_epochs_tanh,train_losses_tanh,label="train loss")
	plot!(training_epochs_tanh,val_losses_tanh,label="validation loss")
	xlabel!("Epoch")
	ylabel!("MSE")
end

# ╔═╡ 4140d5be-0e60-4303-af13-a574efcc2467
md"""
### Much smoother output due to tanh activation. One can see that the tails of the model prediction resemble tanh function.
"""

# ╔═╡ 92b5af78-257f-434f-ada9-e3e17c0cfed7
md"""
We now test the performance of the tanh model here. You can modify the code used to check performance of relu model.
```
let
	#Your code here.
end
```
"""

# ╔═╡ 3a693100-5a3c-44a5-9953-9f64be37370d
let 
	sorted_x_test_batch = reshape(sorted_x_test_vector,1,:) #Reshape the sorted_x_test_vector here
	y_pred = vec(tanh_model(sorted_x_test_batch))  
	plot(title="Testing output for ReLU model")
	scatter!(vec(x_test_batch),vec(y_test_batch),label="Data")
	plot!(sorted_x_test_vector,y_pred,label="Prediction")
end

# ╔═╡ e709d361-9a8c-4c5c-82e3-07b1a36fab58
md"""
## Visualization of activation functions in the hidden layers
"""

# ╔═╡ 8be54f72-cccd-4f84-ab13-6564a287233f
md"""
Here we will now see how the activation functions modify the output from the neurons before passing it to next layer in the network. We first start with looking at the first hidden layer. Observe the individual ReLUs from your model. Each neuron linearizes our input x before passing it to the next layer.
```
let
	x_eval = Float32.(range(x_low, x_up, length=100))
	x_in = reshape(x_eval, 1, :) #Converting it into shape accepted by the input layer of the network
	a1 = model[1](x_in) #Implementing the first hidden layer on the input dataset.
	n_neurons = size(model[1].weight,1)
	plot(title="Output of layer 1 for ReLU model")
	plot!(x_eval,vec(a1[1,:]),label="Neuron 1")
	for neuron in 2:n_neurons-1
		plot!(x_eval,vec(a1[neuron,:]),label="Neuron $(neuron)")
	end
	plot!(x_eval,vec(a1[n_neurons,:]),label="Neuron $(n_neurons)")
	xlabel!("x")
	ylabel!("Output of neurons")
end
```
"""

# ╔═╡ d5693346-ebec-4b8a-bef3-edc4d77ce313
let
	x_eval = Float32.(range(x_low, x_up, length=100))
	x_in = reshape(x_eval, 1, :) #Converting it into shape accepted by the input layer of the network
	a1 = relu_model[1](x_in) #Implementing the first hidden layer on the input dataset.
	n_neurons = size(relu_model[1].weight,1)
	plot(title="Output of layer 1 for ReLU model")
	plot!(x_eval,vec(a1[1,:]),label="Neuron 1")
	for neuron in 2:n_neurons-1
		plot!(x_eval,vec(a1[neuron,:]),label="Neuron $(neuron)")
	end
	plot!(x_eval,vec(a1[n_neurons,:]),label="Neuron $(n_neurons)")
	xlabel!("x")
	ylabel!("Output of neurons")
end

# ╔═╡ 8640909c-8186-4a44-b934-7640a6fce9e2
md"""
Next, we look at the output for second hidden layer. Can you identify the individual relu functions?
```
let
	x_eval = Float32.(range(x_low, x_up, length=100))
	x_in = reshape(x_eval, 1, :)
	a1 = model[1](x_in) #Output of first hidden layer
	a2 = model[2](a1) #Output of first hidden layer is input for second hidden layer
	n_neurons = size(model[2].weight,1)
	plot(title="Output of layer 2 for ReLU model")
	plot!(x_eval,vec(a2[1,:]),label="Neuron 1")
	for neuron in 2:n_neurons-1
		plot!(x_eval,vec(a2[neuron,:]),label="Neuron $(neuron)")
	end
	plot!(x_eval,vec(a2[n_neurons,:]),label="Neuron $(n_neurons)")
	xlabel!("x")
	ylabel!("Output of neurons")
end
```
"""

# ╔═╡ 40c76d00-0165-426f-acc5-7120c621c01d
let
	x_eval = Float32.(range(x_low, x_up, length=100))
	x_in = reshape(x_eval, 1, :)
	a1 = relu_model[1](x_in) #Output of first hidden layer
	a2 = relu_model[2](a1) #Output of first hidden layer is input for second hidden layer
	n_neurons = size(relu_model[2].weight,1)
	plot(title="Output of layer 2 for ReLU model")
	plot!(x_eval,vec(a2[1,:]),label="Neuron 1")
	for neuron in 2:n_neurons-1
		plot!(x_eval,vec(a2[neuron,:]),label="Neuron $(neuron)")
	end
	plot!(x_eval,vec(a2[n_neurons,:]),label="Neuron $(n_neurons)")
	xlabel!("x")
	ylabel!("Output of neurons")
end

# ╔═╡ 07e7475e-314a-450d-8ed8-d5a71b502cae
md"""
### Below, write down a plotting script to plot output of last layer and convince yourself that it resembles the prediction of the polynomial by the model
"""

# ╔═╡ e1b984e1-c58c-4a7f-a515-a746cba61f01
#Code here :)
let
	x_eval = Float32.(range(x_low, x_up, length=100))
	x_in = reshape(x_eval, 1, :)
	a1 = relu_model[1](x_in) #Output of first hidden layer
	a2 = relu_model[2](a1) #Output of first hidden layer is input for second hidden layer
	a3 = relu_model[3](a2)
	n_neurons = size(relu_model[3].weight,1)
	plot(title="Output of layer 3 for ReLU model")
	plot!(x_eval,vec(a3[1,:]),label="Neuron 1")
	xlabel!("x")
	ylabel!("Output of neurons")
end

# ╔═╡ c79c58dc-5d8f-41a6-bcda-c9cd130ac0d3
md"""
### Now, we visualize the activations for the tanh model. Modify the code used for relu model to plot the output of hidden layers for the tanh model.
"""

# ╔═╡ 7d0a5cde-8b6d-4545-a652-e70bdf90e1d9
#Code here for hidden layer 1
let
	x_eval = Float32.(range(x_low, x_up, length=100))
	x_in = reshape(x_eval, 1, :) #Converting it into shape accepted by the input layer of the network
	a1 = tanh_model[1](x_in) #Implementing the first hidden layer on the input dataset.
	n_neurons = size(tanh_model[1].weight,1)
	plot(title="Output of layer 1 for tanh model")
	plot!(x_eval,vec(a1[1,:]),label="Neuron 1")
	for neuron in 2:n_neurons-1
		plot!(x_eval,vec(a1[neuron,:]),label="Neuron $(neuron)")
	end
	plot!(x_eval,vec(a1[n_neurons,:]),label="Neuron $(n_neurons)")
	xlabel!("x")
	ylabel!("Output of neurons")
end

# ╔═╡ f8ec8970-d8fa-4caf-b84d-41b83d4fc7ca
#Code here for hidden layer 2
let
	x_eval = Float32.(range(x_low, x_up, length=100))
	x_in = reshape(x_eval, 1, :)
	a1 = tanh_model[1](x_in) #Output of first hidden layer
	a2 = tanh_model[2](a1) #Output of first hidden layer is input for second hidden layer
	n_neurons = size(tanh_model[2].weight,1)
	plot(title="Output of layer 2 for tanh model")
	plot!(x_eval,vec(a2[1,:]),label="Neuron 1")
	for neuron in 2:n_neurons-1
		plot!(x_eval,vec(a2[neuron,:]),label="Neuron $(neuron)")
	end
	plot!(x_eval,vec(a2[n_neurons,:]),label="Neuron $(n_neurons)")
	xlabel!("x")
	ylabel!("Output of neurons")
end

# ╔═╡ 83f63a7f-cbed-4360-81b3-e80f60aca990
#Code here for final output layer
let
	x_eval = Float32.(range(x_low, x_up, length=100))
	x_in = reshape(x_eval, 1, :)
	a1 = tanh_model[1](x_in) #Output of first hidden layer
	a2 = tanh_model[2](a1) #Output of first hidden layer is input for second hidden layer
	a3 = tanh_model[3](a2)
	n_neurons = size(relu_model[3].weight,1)
	plot(title="Output of layer 3 for ReLU model")
	plot!(x_eval,vec(a3[1,:]),label="Neuron 1")
	xlabel!("x")
	ylabel!("Output of neurons")
end

# ╔═╡ Cell order:
# ╠═771d7bf5-f25f-414a-924b-e6c47cb7afd7
# ╠═a2057ae6-869f-4dea-b388-5a885325c395
# ╠═215cd2be-7fe2-4e04-895f-248d82d0ef8e
# ╠═50fda67c-5d19-47ee-a993-13951ae20076
# ╠═6f832b95-03a1-498b-bd9d-a9f0f0d7bd2c
# ╠═8b97ec1f-303c-46b6-b9d3-1768c9d46cf4
# ╠═ba452564-4d14-4f1f-ad9e-f2c4d86e30ff
# ╠═d4c0bde6-8ec2-4a3a-ac58-c72aa69c15d1
# ╟─2f4e21f5-d3d3-4de4-a692-a53be8cb54fb
# ╟─debc7bf3-aef3-4b29-a2b4-b58fdeb8d806
# ╟─1c0f87e9-ee3b-4a6b-a0de-ca44a35a7366
# ╠═6a7a51de-2544-4719-828f-4d8eb7bb5a38
# ╠═c36237c5-c7ab-411d-b252-6850b5c5f05f
# ╟─a76dbf3e-ed2f-4f50-9a8d-8c072befd1d4
# ╠═9714ca80-59c1-4006-9035-e6b8f83542d2
# ╟─50335286-1c36-4f06-97cc-1dc42c815bc9
# ╟─98d1a8ad-f1d4-41ce-a111-6a0c9c42155c
# ╟─6fb19698-c20f-4399-bd0e-4ee1a61488cf
# ╠═bd009118-542d-423b-94fb-4eeb9d3d8394
# ╟─8a40a81f-8bda-4492-be94-833810dc18c2
# ╠═e9bb0ec2-6d3f-4025-8199-0bd976d0c0ba
# ╟─4ae9326a-ed88-4048-a85b-124d42fe8faf
# ╠═a0d842ee-8740-48f5-b994-9f405763d5d8
# ╠═c810672c-11d9-45c7-8a91-d48e9f8177e0
# ╟─294e5c00-14b8-4e0f-8109-fd1c8335fbc9
# ╠═6f716900-b826-4162-b63b-e62de3ed5d15
# ╟─8515e2da-6d31-4179-a838-2cf6f1a22ad8
# ╟─d6f71020-1678-41cd-b508-8c4c40758349
# ╠═605ee93a-46ce-40d8-8bed-db901fe8372c
# ╟─5ba69755-ba82-441b-b8a5-1b354b262b43
# ╠═4ccdbf9f-b280-4677-a852-f3b3552bd9f5
# ╟─4140d5be-0e60-4303-af13-a574efcc2467
# ╟─92b5af78-257f-434f-ada9-e3e17c0cfed7
# ╠═3a693100-5a3c-44a5-9953-9f64be37370d
# ╟─e709d361-9a8c-4c5c-82e3-07b1a36fab58
# ╠═8be54f72-cccd-4f84-ab13-6564a287233f
# ╠═d5693346-ebec-4b8a-bef3-edc4d77ce313
# ╠═8640909c-8186-4a44-b934-7640a6fce9e2
# ╠═40c76d00-0165-426f-acc5-7120c621c01d
# ╠═07e7475e-314a-450d-8ed8-d5a71b502cae
# ╠═e1b984e1-c58c-4a7f-a515-a746cba61f01
# ╠═c79c58dc-5d8f-41a6-bcda-c9cd130ac0d3
# ╠═7d0a5cde-8b6d-4545-a652-e70bdf90e1d9
# ╠═f8ec8970-d8fa-4caf-b84d-41b83d4fc7ca
# ╠═83f63a7f-cbed-4360-81b3-e80f60aca990
