import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise session 3 : Training neural networks

    In these exercises, you will train a simple neural network using the PyTorch package in Python. You will try to model polynomial functions and try to understand impact of activation functions (we will test relu and tanh, you can try one more activation function of your choice). If time permits, we can also discuss a bit about hyperparameter tuning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Defining our polynomial functions
    """)
    return


@app.function
def f1(x):
    return 0.5 * x**2 + 1.2 * x - 2.5


@app.function
def f2(x):
    return x**3 + 2 * x**2 + 5 * x + 1


@app.function
def f3(x):
    return x**5 + 3 * x**4 - 2.8 * x**3 + 4.9 * x**2 - 0.5 * x + 3


@app.cell
def _():
    # def f4(x):
    #     return ...  # polynomial of your choice
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preparing the sample
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    n_sample : Number of sample points to simulate for datasets
    """)
    return


@app.cell
def _(mo):
    n_sample_slider = mo.ui.slider(
        start=100,
        stop=5000,
        step=100,
        value=1000,
        show_value=True,
        label="n_sample")

    n_sample_slider
    return (n_sample_slider,)


@app.cell
def _(n_sample_slider):
    n_sample = n_sample_slider.value
    return (n_sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Split fractions to filter out training, validation and test datasets from the sample. You can tune train and validation fraction here, test fraction is assumed to be 1 - train fraction - validation fraction.
    """)
    return


@app.cell
def _(mo):
    split_ratio_train_slider = mo.ui.slider(
        start=0.1,
        stop=0.8,
        step=0.1,
        value=0.4,
        show_value=True,
        label="train split fraction")

    split_ratio_val_slider = mo.ui.slider(
        start=0.1,
        stop=0.8,
        step=0.1,
        value=0.4,
        show_value=True,
        label="validation split fraction")

    mo.hstack([split_ratio_train_slider, split_ratio_val_slider])
    return split_ratio_train_slider, split_ratio_val_slider


@app.cell
def _(split_ratio_train_slider, split_ratio_val_slider):
    split_ratio_train = split_ratio_train_slider.value
    split_ratio_val = split_ratio_val_slider.value
    split_ratio_test = 1.0 - split_ratio_train - split_ratio_val

    if split_ratio_test <= 0:
        raise ValueError(
            "train split fraction + validation split fraction must be smaller than 1.")
    return split_ratio_train, split_ratio_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As done yesterday, we will create a sample of random points and try to train a neural network with it to model $y = \mathrm{poly}(x)$ where $\mathrm{poly}(x)$ is a polynomial function (you can use one of the polynomials or define your own one in the available cell). Note that for training the network, it would be ideal to normalize the y values using mean and standard deviation as:

    \[
    y_{norm} = \frac{y-\mu_y}{\sigma_y}
    \]

    Generate the data, calculate mean and standard deviation of the y values, normalize them and split the shuffled sample into training, validation and test datasets.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import torch
    from torch import nn

    return nn, pd, torch


app._unparsable_cell(
    r"""
    # Generate data
    x_low = -2.0
    x_up = 2.0

    _rng = np.random.default_rng(123)
    xs = (x_low + (x_up - x_low) * _rng.random(n_sample)).astype(np.float32)
    ys = (f1(xs) + 0.01 * _rng.standard_normal(n_sample)).astype(np.float32)  # Add noise

    data = pd.DataFrame({"x": xs, "y": ys})

    y_mean =  # Write the mean here
    y_std =  # Write the standard deviation here
    data["ynorm"] = # Write the normalized yvalues here.

    # Split into train and validation sets
    data =  # Shuffle the dataframe
    n_train = int(np.floor(split_ratio_train * n_sample))
    n_val = int(np.floor(split_ratio_val * n_sample))

    x_train = data.loc[, "x"].to_numpy(dtype=np.float32)
    y_train = data.loc[, "ynorm"].to_numpy(dtype=np.float32)

    x_val = data.loc[, "x"].to_numpy(dtype=np.float32)
    y_val = data.loc[, "ynorm"].to_numpy(dtype=np.float32)

    x_test = data.loc[, "x"].to_numpy(dtype=np.float32)
    y_test = data.loc[, "ynorm"].to_numpy(dtype=np.float32)

    # Reshape
    x_train_batch = 
    y_train_batch = 
    x_val_batch = 
    y_val_batch = 
    x_test_batch = 
    y_test_batch = 

    sorted_x_test_vector = np.sort(x_test)  # This we will need later while making predictions
    """,
    name="_"
)


@app.cell
def _(n_sample, np, pd, split_ratio_train, split_ratio_val, torch):
    # Solution
    x_low = -2.0
    x_up = 2.0

    _rng = np.random.default_rng(123)
    xs = (x_low + (x_up - x_low) * _rng.random(n_sample)).astype(np.float32)
    ys = (f1(xs) + 0.01 * _rng.standard_normal(n_sample)).astype(np.float32)  # Add noise

    data = pd.DataFrame({"x": xs, "y": ys})

    y_mean = np.mean(ys)  # Write the mean here
    y_std = np.std(ys, ddof=1)  # Write the standard deviation here
    data["ynorm"] = (ys - y_mean) / y_std  # Write the normalized yvalues here.

    # Split into train and validation sets
    data = data.sample(frac=1, random_state=123).reset_index(drop=True)  # Shuffle the dataframe
    n_train = int(np.floor(split_ratio_train * n_sample))
    n_val = int(np.floor(split_ratio_val * n_sample))

    x_train = data.loc[: n_train - 1, "x"].to_numpy(dtype=np.float32)
    y_train = data.loc[: n_train - 1, "ynorm"].to_numpy(dtype=np.float32)

    x_val = data.loc[n_train : n_train + n_val - 1, "x"].to_numpy(dtype=np.float32)
    y_val = data.loc[n_train : n_train + n_val - 1, "ynorm"].to_numpy(dtype=np.float32)

    x_test = data.loc[n_train + n_val :, "x"].to_numpy(dtype=np.float32)
    y_test = data.loc[n_train + n_val :, "ynorm"].to_numpy(dtype=np.float32)

    x_train_batch = torch.from_numpy(x_train).reshape(-1, 1)
    y_train_batch = torch.from_numpy(y_train).reshape(-1, 1)
    x_val_batch = torch.from_numpy(x_val).reshape(-1, 1)
    y_val_batch = torch.from_numpy(y_val).reshape(-1, 1)
    x_test_batch = torch.from_numpy(x_test).reshape(-1, 1)
    y_test_batch = torch.from_numpy(y_test).reshape(-1, 1)

    sorted_x_test_vector = np.sort(x_test)  # This we will need later while making predictions
    return (
        sorted_x_test_vector,
        x_low,
        x_test,
        x_train,
        x_train_batch,
        x_up,
        x_val,
        x_val_batch,
        y_test,
        y_train,
        y_train_batch,
        y_val,
        y_val_batch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lets visualize our datasets by plotting them (Do not plot the test dataset!)

    What do you expect for the train and validation datasets to look like?
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(plt, x_train, x_val, y_train, y_val):
    fig_data, ax_data = plt.subplots()
    ax_data.set_title("Data distribution")
    ax_data.scatter(x_train, y_train, label="Train Data")
    ax_data.scatter(x_val, y_val, label="Validation Data")
    ax_data.set_xlabel("x")
    ax_data.set_ylabel("y")
    ax_data.legend()
    fig_data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training with relu activation function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    n_epochs : Number of training steps to update the weights
    """)
    return


@app.cell
def _(mo):
    epochs_slider = mo.ui.slider(
        start=100,
        stop=5000,
        step=100,
        value=1000,
        show_value=True,
        label="n_epochs")

    epochs_slider
    return (epochs_slider,)


@app.cell
def _(epochs_slider):
    epochs = epochs_slider.value
    return (epochs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    wait_period : Number of epochs to wait before applying early stopping
    """)
    return


@app.cell
def _(epochs, mo):
    wait_period_slider = mo.ui.slider(
        start=10,
        stop=max(10, epochs),
        step=10,
        value=min(100, epochs),
        show_value=True,
        label="wait_period")

    wait_period_slider
    return (wait_period_slider,)


@app.cell
def _(wait_period_slider):
    wait_period = wait_period_slider.value
    return (wait_period,)


@app.cell
def _():
    learning_rate = 0.05  # Setting the optimizer learning rate. One can also use Adam optimizer.
    return (learning_rate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we come to defining the model and writing the training loop for the neural network. Some things to do:

    1. Visualize the neural network using the architecture defined. You can use Excalib tool to draw it
    2. For starters, let it train for all the epochs and plot the loss curves (for training and validation samples) in the next cell
    3. Implement early stopping criteria

    We use dense layers and currently use the ReLU activation function to define our neural network. The architecture is

    `1 → 8 → 8 → 1`.
    """)
    return


@app.cell
def _(
    epochs,
    learning_rate,
    nn,
    torch,
    wait_period,
    x_train_batch,
    x_val_batch,
    y_train_batch,
    y_val_batch,
):
    # Define neural network model here. We use dense layers and currently use relu activation function to define our neural network.
    torch.manual_seed(123)
    relu_model = nn.Sequential(
        nn.Linear(1, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 1))

    # Track losses
    train_losses = []  # Loss for training dataset
    val_losses = []  # Loss for validation dataset
    training_epochs = []

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(relu_model.parameters(), lr=learning_rate)

    # Calculating initial validation loss, will be used for early stopping
    with torch.no_grad():
        best_val_loss = criterion(relu_model(x_val_batch), y_val_batch).item()

    wait_time = 1  # loop variable for early stopping criteria

    for epoch in range(1, epochs + 1):
        # Gradient step
        relu_model.train()
        optimizer.zero_grad()
        train_prediction = relu_model(x_train_batch)
        loss = criterion(train_prediction, y_train_batch)  # Gradients are calculated from this loss
        loss.backward()
        optimizer.step()  # Weights updated by the optimizer

        # Record losses
        relu_model.eval()
        with torch.no_grad():
            train_loss = criterion(relu_model(x_train_batch), y_train_batch).item()
            val_loss = criterion(relu_model(x_val_batch), y_val_batch).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        training_epochs.append(epoch)

    # Applying early stopping (After training and inspecting the loss curves). We compare the val_loss with the best_val_loss. If they do not differ by some small factor delta, then we increment the wait_time loop variable; else we replace best_val_loss with current val_loss. Once the wait_time reaches wait_period, we terminate training.
        if abs(val_loss - best_val_loss) < 0.001:
            wait_time += 1
        else:
            best_val_loss = val_loss

        if wait_time == wait_period:
            print(
                f"Validation loss hasnt improved much for {wait_period} epochs. "
                f"Stopping the training at epoch {epoch}")
            break

        # Print every 100 epochs
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch} - Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}")
    return relu_model, train_losses, training_epochs, val_losses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once the training is done, lets see how the losses change over training.
    """)
    return


app._unparsable_cell(
    r"""
    fig_relu_loss, ax_relu_loss = plt.subplots()
    ax_relu_loss.set_title("Evolution of loss")
    ax_relu_loss.plot(, , label="train loss")
    ax_relu_loss.plot(, , label="validation loss")
    ax_relu_loss.set_xlabel("Epoch")
    ax_relu_loss.set_ylabel("MSE")
    ax_relu_loss.legend()
    fig_relu_loss
    """,
    name="_"
)


@app.cell
def _(plt, train_losses, training_epochs, val_losses):
    # Solution:
    fig_relu_loss, ax_relu_loss = plt.subplots()
    ax_relu_loss.set_title("Evolution of loss")
    ax_relu_loss.plot(training_epochs, train_losses, label="train loss")
    ax_relu_loss.plot(training_epochs, val_losses, label="validation loss")
    ax_relu_loss.set_xlabel("Epoch")
    ax_relu_loss.set_ylabel("MSE")
    ax_relu_loss.legend()
    fig_relu_loss
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The ReLU function linearizes the output and its effect is visible in the output of the model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lastly, we check the performance of the model on test dataset.
    """)
    return


app._unparsable_cell(
    r"""
    sorted_x_test_batch_relu = torch.from_numpy(sorted_x_test_vector).reshape(-1, 1)

    relu_model.eval()
    with torch.no_grad():
        y_pred_relu = relu_model(sorted_x_test_batch_relu).squeeze(1).numpy()

    fig_relu_test, ax_relu_test = plt.subplots()
    ax_relu_test.set_title("Testing output for ReLU model")
    ax_relu_test.scatter(, , label="Data")
    ax_relu_test.plot(, , label="Prediction")
    ax_relu_test.set_xlabel("x")
    ax_relu_test.set_ylabel("y")
    ax_relu_test.legend()
    fig_relu_test
    """,
    name="_"
)


@app.cell
def _(plt, relu_model, sorted_x_test_vector, torch, x_test, y_test):
    sorted_x_test_batch_relu = torch.from_numpy(sorted_x_test_vector).reshape(-1, 1)

    relu_model.eval()
    with torch.no_grad():
        y_pred_relu = relu_model(sorted_x_test_batch_relu).squeeze(1).numpy()

    fig_relu_test, ax_relu_test = plt.subplots()
    ax_relu_test.set_title("Testing output for ReLU model")
    ax_relu_test.scatter(x_test, y_test, label="Data")
    ax_relu_test.plot(sorted_x_test_vector, y_pred_relu, color="orange", label="Prediction")
    ax_relu_test.set_xlabel("x")
    ax_relu_test.set_ylabel("y")
    ax_relu_test.legend()
    fig_relu_test
    return (y_pred_relu,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training with tanh activation function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we define a model with same architecture as before but with tanh activation function. Complete the training loop below, you can copy paste the missing sections from earlier training loop.
    """)
    return


app._unparsable_cell(
    r"""
    # Define neural network model
    torch.manual_seed(123)
    tanh_model = nn.Sequential() # Define your model here

    # Track loss
    train_losses_tanh = []
    val_losses_tanh = []
    training_epochs_tanh = []

    criterion_tanh = nn.MSELoss()
    optimizer_tanh = torch.optim.SGD(tanh_model.parameters(), lr=learning_rate)

    with torch.no_grad():
        best_val_loss_tanh = criterion_tanh(tanh_model(x_val_batch), y_val_batch).item()

    wait_time_tanh = 1  # loop variable for early stopping criteria

    for epoch_tanh in range(1, epochs + 1):
        # Gradient step


        # Record losses

        # Applying early stopping

        # Print every 100 epochs
    """,
    name="_"
)


@app.cell
def _(
    epochs,
    learning_rate,
    nn,
    torch,
    wait_period,
    x_train_batch,
    x_val_batch,
    y_train_batch,
    y_val_batch,
):
    # Solution:
    torch.manual_seed(123)
    tanh_model = nn.Sequential(
        nn.Linear(1, 8),
        nn.Tanh(),
        nn.Linear(8, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
    )

    # Track loss
    train_losses_tanh = []
    val_losses_tanh = []
    training_epochs_tanh = []

    criterion_tanh = nn.MSELoss()
    optimizer_tanh = torch.optim.SGD(tanh_model.parameters(), lr=learning_rate)

    with torch.no_grad():
        best_val_loss_tanh = criterion_tanh(tanh_model(x_val_batch), y_val_batch).item()

    wait_time_tanh = 1  # loop variable for early stopping criteria

    for epoch_tanh in range(1, epochs + 1):
        # Gradient step
        tanh_model.train()
        optimizer_tanh.zero_grad()
        train_prediction_tanh = tanh_model(x_train_batch)
        loss_tanh = criterion_tanh(train_prediction_tanh, y_train_batch)
        loss_tanh.backward()
        optimizer_tanh.step()

        # Record losses
        tanh_model.eval()
        with torch.no_grad():
            train_loss_tanh = criterion_tanh(tanh_model(x_train_batch), y_train_batch).item()
            val_loss_tanh = criterion_tanh(tanh_model(x_val_batch), y_val_batch).item()

        train_losses_tanh.append(train_loss_tanh)
        val_losses_tanh.append(val_loss_tanh)
        training_epochs_tanh.append(epoch_tanh)

        # Applying early stopping
        if abs(val_loss_tanh - best_val_loss_tanh) < 0.001:
            wait_time_tanh += 1
        else:
            best_val_loss_tanh = val_loss_tanh

        if wait_time_tanh == wait_period:
            print(
                f"Validation loss hasnt improved much for {wait_period} epochs. "
                f"Stopping the training at epoch {epoch_tanh}"
            )
            break

        # Print every 100 epochs
        if epoch_tanh % 100 == 0:
            print(
                f"Epoch {epoch_tanh} - Train Loss: {train_loss_tanh:.4f}, "
                f"Val Loss: {val_loss_tanh:.4f}"
            )
    return tanh_model, train_losses_tanh, training_epochs_tanh, val_losses_tanh


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot the loss values now for tanh model.
    """)
    return


@app.cell
def _():
    # Write here
    return


@app.cell
def _(plt, train_losses_tanh, training_epochs_tanh, val_losses_tanh):
    # Solution:
    fig_tanh_loss, ax_tanh_loss = plt.subplots()
    ax_tanh_loss.set_title("Evolution of loss for tanh model")
    ax_tanh_loss.plot(training_epochs_tanh, train_losses_tanh, label="train loss")
    ax_tanh_loss.plot(training_epochs_tanh, val_losses_tanh, label="validation loss")
    ax_tanh_loss.set_xlabel("Epoch")
    ax_tanh_loss.set_ylabel("MSE")
    ax_tanh_loss.legend()
    fig_tanh_loss
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Much smoother output due to tanh activation. One can see that the tails of the model prediction resemble tanh function.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now test the performance of the tanh model here. You can modify the code used to check performance of relu model.
    """)
    return


@app.cell
def _():
    # Write here
    return


@app.cell
def _(plt, sorted_x_test_vector, tanh_model, torch, x_test, y_test):
    # Solution
    sorted_x_test_batch_tanh = torch.from_numpy(sorted_x_test_vector).reshape(-1, 1)

    tanh_model.eval()
    with torch.no_grad():
        y_pred_tanh = tanh_model(sorted_x_test_batch_tanh).squeeze(1).numpy()

    fig_tanh_test, ax_tanh_test = plt.subplots()
    ax_tanh_test.set_title("Testing output for tanh model")
    ax_tanh_test.scatter(x_test, y_test, label="Data")
    ax_tanh_test.plot(sorted_x_test_vector, y_pred_tanh, color="orange", label="Prediction")
    ax_tanh_test.set_xlabel("x")
    ax_tanh_test.set_ylabel("y")
    ax_tanh_test.legend()
    fig_tanh_test
    return (y_pred_tanh,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualization of activation functions in the hidden layers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we will now see how the activation functions modify the output from the neurons before passing it to next layer in the network. We first start with looking at the first hidden layer. Observe the individual ReLUs from your model. Each neuron linearizes our input x before passing it to the next layer.
    """)
    return


@app.cell
def _(np, torch, x_low, x_up):
    x_eval = np.linspace(x_low, x_up, 100, dtype=np.float32)
    x_in = torch.from_numpy(x_eval).reshape(-1, 1)  # Converting it into shape accepted by the input layer of the network
    return x_eval, x_in


@app.cell
def _(plt, relu_model, torch, x_eval, x_in):
    relu_model.eval()
    with torch.no_grad():
        a1_relu = relu_model[1](relu_model[0](x_in)).numpy()  # Implementing the first hidden layer on the input dataset.

    n_neurons_relu_1 = a1_relu.shape[1]

    fig_relu_layer1, ax_relu_layer1 = plt.subplots()
    ax_relu_layer1.set_title("Output of layer 1 for ReLU model")
    for neuron_relu_1 in range(n_neurons_relu_1):
        ax_relu_layer1.plot(x_eval, a1_relu[:, neuron_relu_1], label=f"Neuron {neuron_relu_1 + 1}")
    ax_relu_layer1.set_xlabel("x")
    ax_relu_layer1.set_ylabel("Output of neurons")
    ax_relu_layer1.legend()
    fig_relu_layer1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we look at the output for second hidden layer. Can you identify the individual relu functions?
    """)
    return


@app.cell
def _(plt, relu_model, torch, x_eval, x_in):
    relu_model.eval()
    with torch.no_grad():
        _a1_relu = relu_model[1](relu_model[0](x_in))  # Output of first hidden layer
        a2_relu = relu_model[3](relu_model[2](_a1_relu)).numpy()  # Output of first hidden layer is input for second hidden layer

    n_neurons_relu_2 = a2_relu.shape[1]

    fig_relu_layer2, ax_relu_layer2 = plt.subplots()
    ax_relu_layer2.set_title("Output of layer 2 for ReLU model")
    for neuron_relu_2 in range(n_neurons_relu_2):
        ax_relu_layer2.plot(x_eval, a2_relu[:, neuron_relu_2], label=f"Neuron {neuron_relu_2 + 1}")
    ax_relu_layer2.set_xlabel("x")
    ax_relu_layer2.set_ylabel("Output of neurons")
    ax_relu_layer2.legend()
    fig_relu_layer2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below, write down a plotting script to plot output of last layer and convince yourself that it resembles the prediction of the polynomial by the model
    """)
    return


app._unparsable_cell(
    r"""
    relu_model.eval()
    with torch.no_grad():


    fig_relu_layer3, ax_relu_layer3 = plt.subplots()
    ax_relu_layer3.set_title("Output of layer 3 for ReLU model")
    ax_relu_layer3.plot(, , label="Neuron 1")
    ax_relu_layer3.plot(sorted_x_test_vector, y_pred_relu, color="orange", label="Prediction")
    ax_relu_layer3.set_xlabel("x")
    ax_relu_layer3.set_ylabel("Output of neurons")
    ax_relu_layer3.legend()
    fig_relu_layer3
    """,
    name="_"
)


@app.cell
def _(plt, relu_model, sorted_x_test_vector, torch, x_eval, x_in, y_pred_relu):
    # Solution
    relu_model.eval()
    with torch.no_grad():
        _a1_relu_final = relu_model[1](relu_model[0](x_in))
        _a2_relu_final = relu_model[3](relu_model[2](_a1_relu_final))
        a3_relu = relu_model[4](_a2_relu_final).squeeze(1).numpy()

    fig_relu_layer3, ax_relu_layer3 = plt.subplots()
    ax_relu_layer3.set_title("Output of layer 3 for ReLU model")
    ax_relu_layer3.plot(x_eval, a3_relu, label="Neuron 1")
    ax_relu_layer3.plot(sorted_x_test_vector, y_pred_relu, color="orange", label="Prediction")
    ax_relu_layer3.set_xlabel("x")
    ax_relu_layer3.set_ylabel("Output of neurons")
    ax_relu_layer3.legend()
    fig_relu_layer3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we visualize the activations for the tanh model. Modify the code used for relu model to plot the output of hidden layers for the tanh model.
    """)
    return


@app.cell
def _(plt, tanh_model, torch, x_eval, x_in):
    # Solution
    tanh_model.eval()
    with torch.no_grad():
        a1_tanh = tanh_model[1](tanh_model[0](x_in)).numpy()

    n_neurons_tanh_1 = a1_tanh.shape[1]

    fig_tanh_layer1, ax_tanh_layer1 = plt.subplots()
    ax_tanh_layer1.set_title("Output of layer 1 for tanh model")
    for neuron_tanh_1 in range(n_neurons_tanh_1):
        ax_tanh_layer1.plot(x_eval, a1_tanh[:, neuron_tanh_1], label=f"Neuron {neuron_tanh_1 + 1}")
    ax_tanh_layer1.set_xlabel("x")
    ax_tanh_layer1.set_ylabel("Output of neurons")
    ax_tanh_layer1.legend()
    fig_tanh_layer1
    return


@app.cell
def _(plt, tanh_model, torch, x_eval, x_in):
    # Solution
    tanh_model.eval()
    with torch.no_grad():
        _a1_tanh = tanh_model[1](tanh_model[0](x_in))
        a2_tanh = tanh_model[3](tanh_model[2](_a1_tanh)).numpy()

    n_neurons_tanh_2 = a2_tanh.shape[1]

    fig_tanh_layer2, ax_tanh_layer2 = plt.subplots()
    ax_tanh_layer2.set_title("Output of layer 2 for tanh model")
    for neuron_tanh_2 in range(n_neurons_tanh_2):
        ax_tanh_layer2.plot(x_eval, a2_tanh[:, neuron_tanh_2], label=f"Neuron {neuron_tanh_2 + 1}")
    ax_tanh_layer2.set_xlabel("x")
    ax_tanh_layer2.set_ylabel("Output of neurons")
    ax_tanh_layer2.legend()
    fig_tanh_layer2
    return


@app.cell
def _(plt, sorted_x_test_vector, tanh_model, torch, x_eval, x_in, y_pred_tanh):
    # Solution
    tanh_model.eval()
    with torch.no_grad():
        _a1_tanh_final = tanh_model[1](tanh_model[0](x_in))
        _a2_tanh_final = tanh_model[3](tanh_model[2](_a1_tanh_final))
        a3_tanh = tanh_model[4](_a2_tanh_final).squeeze(1).numpy()

    fig_tanh_layer3, ax_tanh_layer3 = plt.subplots()
    ax_tanh_layer3.set_title("Output of layer 3 for tanh model")
    ax_tanh_layer3.plot(x_eval, a3_tanh, label="Neuron 1")
    ax_tanh_layer3.plot(sorted_x_test_vector, y_pred_tanh, color="orange", label="Prediction")
    ax_tanh_layer3.set_xlabel("x")
    ax_tanh_layer3.set_ylabel("Output of neurons")
    ax_tanh_layer3.legend()
    fig_tanh_layer3
    return


if __name__ == "__main__":
    app.run()
