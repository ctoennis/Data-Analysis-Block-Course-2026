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
    # Regression trees, random forests, and loss functions

    In this exercise session, you will learn:

    1. How to train regression trees for 1D and multivariate functions.
    2. How to train random forests and compare them with single trees.
    3. How the Huber loss differs from a squared-error loss.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 1: Regression Trees

    A regression tree recursively splits the data into two until the maximum number of splits is reached. As a first exercise, we will play along with some 1D and 2D functions and use `scikit.learn` to train a regression tree. We use `pandas` to store data in data-frames
    """)
    return


@app.cell
def _():
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor, export_text

    return DecisionTreeRegressor, export_text, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.1 Functions and controls

    Simulated data of some complex functions with some gaussian noise added to it. A part of this data
    will be used for training a regression tree. Some functions are listed below, you can also write your
    own function
    """)
    return


@app.cell
def _(np):
    def f1_1D(x):
        return np.sin(2*x)

    def f2_1D(x):
        return np.sin(np.exp(-(x**2)))

    def f3_1D(x):
        return 2 * np.exp(-2*x)

    return


@app.cell
def _(mo):
    n_sample_slider = mo.ui.slider(
        start=100,
        stop=5000,
        step=100,
        value=1000,
        show_value=True,
        label="Number of simulated data points")

    train_fraction_slider = mo.ui.slider(
        start=0.1,
        stop=0.9,
        step=0.1,
        value=0.5,
        show_value=True,
        label="Training fraction")

    tree_depth_slider = mo.ui.slider(
        start=1,
        stop=20,
        step=1,
        value=5,
        show_value=True,
        label="Maximum tree depth")

    mo.vstack(
        [n_sample_slider,
        train_fraction_slider,
        tree_depth_slider])
    return n_sample_slider, train_fraction_slider, tree_depth_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lets create our sample which will be used to create a decision tree model. We first randomly sample
    input features based on the number of data points (`n_sample`) and use one of the functions above for the labels `y`. Add
    """)
    return


@app.cell
def _(n_sample_slider, train_fraction_slider, tree_depth_slider):
    n_sample = int(n_sample_slider.value)
    train_fraction = float(train_fraction_slider.value)
    tree_depth = int(tree_depth_slider.value)
    return n_sample, train_fraction, tree_depth


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Todo

    Create a noisy sample for the 1D function, store it in a dataframe, shuffle it reproducibly,
    and split it into training and test data.
    """)
    return


@app.cell
def _(data, n_sample, np, pd, train_fraction):
    # Write here
    x_1d = 2 * np.pi * np.random.rand(n_sample)
    y_1d = 'your function'(x_1d) + 0.01 * np.random.standard_normal(n_sample)

    data_1d = pd.DataFrame({"x_data": x_1d, "y_data": y_1d})
    data_1d = data.sample(frac=1.0, random_state=123).reset_index(drop=True) # The random_state guarantees that we always take the same seed of the random number generator

    n_train_1d = int(np.floor(n_sample * train_fraction))
    return data_1d, n_train_1d


@app.cell
def _(data_1d, n_train_1d):
    ex1d_training = data_1d.iloc[:n_train_1d].copy()
    ex1d_test = data_1d.iloc[n_train_1d:].copy()
    return ex1d_test, ex1d_training


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Fit a 1D regression tree

    Scikit-learn expects the feature matrix to have shape `(n_samples, n_features)`. For a single
    feature we therefore reshape the `x`-values to `(N, 1)`.
    """)
    return


@app.cell
def _(DecisionTreeRegressor, ex1d_training, tree_depth):
    features_1d = ex1d_training["x_data"].to_numpy().reshape(-1, 1)
    ytrue_1d = ex1d_training["y_data"].to_numpy()

    model_1d = DecisionTreeRegressor(max_depth=tree_depth, random_state=123)
    model_1d.fit(features_1d, ytrue_1d)
    return features_1d, model_1d, ytrue_1d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lets see the mean squared error for this tree
    """)
    return


@app.cell
def _(ex1d_test, model_1d, np):
    import matplotlib.pyplot as plt

    test_features_1d = ex1d_test["x_data"].to_numpy().reshape(-1, 1)
    y_pred_1d = model_1d.predict(test_features_1d)
    test_mse_1d = np.mean((y_pred_1d - ex1d_test["y_data"].to_numpy()) ** 2)

    print(f"Test MSE: {test_mse_1d:.6g}")

    fig_1d_test, ax_1d_test = plt.subplots()

    ax_1d_test.scatter(ex1d_test["x_data"], ex1d_test["y_data"], label="test data")
    ax_1d_test.scatter(ex1d_test["x_data"], y_pred_1d, label="model predictions")

    ax_1d_test.set_title("Testing regression tree on test data")
    ax_1d_test.set_xlabel("x_input")
    ax_1d_test.set_ylabel("y_output")
    ax_1d_test.legend()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the step-like output of the model. A shallow tree has only a few leaves, so a whole
    interval of input values receives the same prediction. Increasing the depth gives a finer
    approximation, but very deep trees can overfit the training data.

    You can view your trained tree by printing it out. Notice the splits observed at the leaf nodes based on the features
    """)
    return


@app.cell
def _(export_text, model_1d):
    print(export_text(model_1d, feature_names=["x"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.3 Multivariate regression tree

    We now introduce a target `z = f(x, y)` with two input features.
    """)
    return


@app.cell
def _(np):
    def f1_2D(x, y):
        return np.sin(x) * np.cos(y)

    def f2_2D(x, y):
        return np.sin(x) * np.cos(y**2)

    def f3_2D(x, y):
        return np.log(np.abs(x * np.sin(y)))

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Todo

    As in 1D case: Generate a noisy 2D sample, store it in a dataframe, shuffle it, and create training and test
    datasets.
    """)
    return


app._unparsable_cell(
    r"""
    x_2d = 
    y_2d = 
    z_2d = "your function"(x_2d, y_2d) + 0.005 * np.random.standard_normal(n_sample)

    data_2d = 
    data_2d = 

    n_train_2d = int(np.floor(n_sample * train_fraction))
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot the underlying 2D function. A regular grid is used for the surface so that the shape is
    easier to see than with the randomly sampled training points alone.
    """)
    return


@app.cell
def _(np, plt):
    # Write here
    surface_axis = np.linspace(1.0, 1.0 + 2 * np.pi, 100)
    surface_x, surface_y = np.meshgrid(surface_axis, surface_axis)
    surface_z = "your function"(surface_x, surface_y)

    fig_2d_function = plt.figure()
    ax_2d_function = fig_2d_function.add_subplot(111, projection="3d")
    ax_2d_function.plot_surface(surface_x, surface_y, surface_z, alpha=0.8, cmap="viridis")
    ax_2d_function.set_title("2D function to model")
    ax_2d_function.set_xlabel("x")
    ax_2d_function.set_ylabel("y")
    ax_2d_function.set_zlabel("F(x, y)")

    fig_2d_function
    return


@app.cell
def _(DecisionTreeRegressor, data_2d, n_train_2d, tree_depth):
    ex2d_training = data_2d.iloc[:n_train_2d].copy()
    ex2d_test = data_2d.iloc[n_train_2d:].copy()

    features_2d = ex2d_training[["x_data", "y_data"]].to_numpy()
    ztrue_2d = ex2d_training["z_data"].to_numpy()

    model_2d = DecisionTreeRegressor(max_depth=tree_depth, random_state=123)
    model_2d.fit(features_2d, ztrue_2d)
    return features_2d, model_2d, ztrue_2d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Do you think you need more depth to fit a multivariate function? How does a splitting happens in multivariate functions? See below, what happens
    """)
    return


@app.cell
def _(export_text, model_2d):
    print(export_text(model_2d, feature_names=["x", "y"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Todo

    We test the predictions of our tree on the test data. Use the code from the 1D example
    """)
    return


app._unparsable_cell(
    r"""
    # Write here
    test_features_2d = 
    z_pred_tree_test = 
    test_mse_2d = 

    print(f"test_mse_2d: {test_mse_2d}")

    fig_2d_test = plt.figure()
    ax_2d_test = fig_2d_test.add_subplot(111, projection="3d")

    ax_2d_test.plot_surface(surface_x, surface_y, surface_z, cmap="viridis", alpha=0.15)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At a leaf, randomly only one of the features is picked for splitting. Hence, a coarser tree is expected to
    perform worse if you have more than one feature in your data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2: Random Forest

    You might have noticed how a single tree with lower depth results in coarser representations of our functions which results in higher variance. In addition, if the depth is increased, the tree has a chance to be overfitted. Hence, we can use bagging techniques like random forest in order to optimize the models. A forest is
    1. Group of independent trees
    2. Trained on different subsets of data

    So unlike a single tree, a test input passes through all the trees in the forest and the average of the individual tree outputs is considered as the final prediction of the forest. Lets revisit our 1D example and this time we fit a forest model for it.
    """)
    return


@app.cell
def _(mo):
    n_trees_slider = mo.ui.slider(
        start=10,
        stop=100,
        step=10,
        value=20,
        show_value=True,
        label="Number of trees in the forest")

    n_trees_slider
    return (n_trees_slider,)


@app.cell
def _(n_trees_slider):
    n_trees = int(n_trees_slider.value)
    return (n_trees,)


@app.cell
def _(features_1d, n_trees, tree_depth, ytrue_1d):
    from sklearn.ensemble import RandomForestRegressor

    forest_1d = RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=tree_depth,
        random_state=123,
        n_jobs=-1)

    forest_1d.fit(features_1d, ytrue_1d)
    return (RandomForestRegressor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Todo

    Lets compare the behavior of our decision tree and forest models. Complete the code below and run it
    in the cell below
    """)
    return


app._unparsable_cell(
    r"""
    # Write here
    x_compare_1d = np.arange(0.0, 6.0 + 0.025, 0.05)
    x_features_compare_1d = # reshape
    y_output_compare_1d = "your function"(x_compare_1d)

    y_pred_tree_compare_1d = # predict
    mse_tree_1d = 

    y_pred_forest_compare_1d = 
    mse_forest_1d = 

    print(f"MSE from single tree: {mse_tree_1d}")
    print(f"MSE from forest: {mse_forest_1d}")

    fig_compare_1d, ax_compare_1d = plt.subplots()

    ax_compare_1d.plot(x_compare_1d, y_output_compare_1d, label="Function")
    ax_compare_1d.scatter(x_compare_1d, y_pred_forest_compare_1d, alpha=0.4, label="Forest",)
    ax_compare_1d.scatter(x_compare_1d, y_pred_tree_compare_1d, alpha=0.4, label="Single tree",)

    ax_compare_1d.set_title(f"Max depth per tree = {tree_depth}, N trees = {n_trees}")
    ax_compare_1d.set_xlabel("x_input")
    ax_compare_1d.set_ylabel("y_output")
    ax_compare_1d.legend()
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Random forest for the 2D function

    We come back to the 2D example and try fitting a random forest.
    """)
    return


@app.cell
def _(RandomForestRegressor, features_2d, n_trees, tree_depth, ztrue_2d):
    forest_2d = RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=tree_depth,
        random_state=123,
        n_jobs=-1)

    forest_2d.fit(features_2d, ztrue_2d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Todo

    Compare the 2D single tree and random forest along the line `x = y` for values between 1 and 6.
    """)
    return


app._unparsable_cell(
    r"""
    # Write here
    x_compare_2d = np.arange(1.0, 6.0, 0.05)
    y_compare_2d = np.arange(1.0, 6.0, 0.05)
    xy_features_compare_2d = np.column_stack([x_compare_2d, y_compare_2d])
    z_output_compare_2d = "your function"(x_compare_2d, y_compare_2d)

    z_pred_tree_compare_2d = 
    mse_tree_2d = 

    z_pred_forest_compare_2d = 
    mse_forest_2d = 

    print(f"MSE from single tree: {mse_tree_2d}")
    print(f"MSE from forest: {mse_forest_2d:.6g}")

    fig_compare_2d = plt.figure()
    ax_compare_2d = fig_compare_2d.add_subplot(111, projection="3d")

    ax_compare_2d.plot(x_compare_2d, y_compare_2d, z_output_compare_2d, label="Function")
    ax_compare_2d.scatter(x_compare_2d, y_compare_2d, z_pred_forest_compare_2d, label="Forest")
    ax_compare_2d.scatter(x_compare_2d, y_compare_2d, z_pred_tree_compare_2d, label="Single tree")

    ax_compare_2d.set_title(f"Max depth per tree = {tree_depth}, N trees = {n_trees}")
    ax_compare_2d.set_xlabel("x_input")
    ax_compare_2d.set_ylabel("y_input")
    ax_compare_2d.set_zlabel("z_output")
    ax_compare_2d.legend()
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Notes

    1. A random forest often generalizes better than a single tree because averaging reduces variance.
    2. Very sharp structures can be smoothed by the averaging in a forest, so the best model still
       depends on the target function and the chosen hyperparameters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 3: Loss functions

    Previously, we have only worked with the mean squared error (MSE). A Squared error penalizes large residuals quadratically. The Huber loss behaves quadratically only close to zero and linearly for sufficiently large residuals, making it less sensitive to outliers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Todo

    Implement the Huber loss

    \[
    L_\delta(r) =
    \begin{cases}
    \frac{1}{2}r^2, & |r| \leq \delta,\\
    \delta\left(|r| - \frac{1}{2}\delta\right), & |r| > \delta.
    \end{cases}
    \]
    """)
    return


app._unparsable_cell(
    r"""
    # Write here
    def huber_loss(delta_y, delta=0.5)
        return
    """,
    name="_"
)


@app.cell
def _(mo):
    delta_slider = mo.ui.slider(
        start=0.001,
        stop=0.05,
        step=0.001,
        value=0.001,
        show_value=True,
        label="Huber threshold δ")

    delta_slider
    return (delta_slider,)


@app.cell
def _(delta_slider):
    delta = float(delta_slider.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Todo

    Compare squared error and Huber loss for the residuals of the 1D random-forest prediction.
    """)
    return


app._unparsable_cell(
    r"""
    # Write here
    x_loss = np.arange(0.0, 8.0, 0.01)
    x_features_loss = 
    y_output_loss = "your function"(x_loss)
    y_pred_loss = forest_1d.predict(x_features_loss)

    delta_y = np.sort(y_output_loss - y_pred_loss)
    mse_loss = delta_y**2
    huber_values = np.array([huber_loss(residual, delta=delta) for residual in delta_y])

    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(delta_y, mse_loss, label="MeanSquaredError")
    ax_loss.plot(delta_y, huber_values, label=f"Huber Loss (δ={delta:g})")
    ax_loss.set_title("Comparison of MSE vs Huber loss")
    ax_loss.set_xlabel(r"$y_{true} - y_{prediction}$")
    ax_loss.set_ylabel("Loss value")
    ax_loss.legend()
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For large errors, Huber loss linearizes the cost function
    which results in a lower penalty.
    """)
    return


if __name__ == "__main__":
    app.run()
