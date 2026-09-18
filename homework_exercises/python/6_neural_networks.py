import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import uproot
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.utils import resample
    from sklearn.preprocessing import StandardScaler

    return StandardScaler, mo, nn, np, optim, pd, plt, tf, torch, uproot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Neural Netwworks

    We will look into neural networks on paper a little first:

    Let's take a number of patients in a hospital. We measure their temperature and want to figure out if tey are sick or not using neural netorks. Sick patients have a either too high or too low temperature, healthy patients have a normal temperature.
    """)
    return


@app.cell
def _(np):
    healthy = np.random.normal(36,1,10)
    fever = np.random.normal(38,1,5) 
    hypothermia = np.random.normal(36,1,5)

    print(healthy)
    print(fever)
    print(hypothermia)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we do the following:

    1. Define linear functions to identify the sick populations (activation function)
    2. Make a linear combination of these to identify both populations (end node)
    3. Add functions to identify more of the sick patients (overtraining)
    """)
    return


@app.cell
def _(uproot):
    dfbkg = None
    dfsig = None
    dftest = None

    with uproot.open("background.root") as file:

        tbkg = file["t"]
        dfbkg = tbkg.arrays(library="pd")

    with uproot.open("signal.root") as file:

        tsig = file["t"]
        dfsig = tsig.arrays(library="pd")

    with uproot.open("test.root") as file:

        ttest = file["t"]
        dftest = ttest.arrays(library="pd")
    return dfbkg, dfsig, dftest


@app.cell
def _():
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
    return (log_transform,)


@app.function
def split_dataframe(df, fraction, random_state=None):
    """
    Randomly split a DataFrame into two parts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to split.
    fraction : float
        Fraction of entries to put in the first DataFrame.
        Must be between 0 and 1.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    df1, df2 : pd.DataFrame
        The two resulting DataFrames.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")

    df_shuffled = df.sample(frac=1, random_state=random_state)

    split_index = int(len(df) * fraction)

    df1 = df_shuffled.iloc[:split_index]
    df2 = df_shuffled.iloc[split_index:]

    return df1, df2


@app.cell
def _(np):
    def log_transformer(df, columns):
        df = df.copy()

        for column in columns:
            df[column] = np.log(df[column])

        return df

    return (log_transformer,)


@app.cell
def _(pd):
    def trim_to_common_support(dataframes, columns):
        """
        Trim multiple DataFrames so that every specified column
        has the same min/max range across all DataFrames.
        """
        common_min = {
            col: max(df[col].min() for df in dataframes)
            for col in columns
        }

        common_max = {
            col: min(df[col].max() for df in dataframes)
            for col in columns
        }

        for col in columns:
            if common_min[col] > common_max[col]:
                raise ValueError(f"No common support for column '{col}'.")

        trimmed_dataframes = []

        for df in dataframes:
            mask = pd.Series(True, index=df.index)

            for col in columns:
                mask &= (
                    (df[col] >= common_min[col]) &
                    (df[col] <= common_max[col])
                )

            trimmed_dataframes.append(df.loc[mask].copy())

        return trimmed_dataframes

    return (trim_to_common_support,)


@app.cell
def _(
    dfbkg,
    dfsig,
    dftest,
    log_transform,
    log_transformer,
    trim_to_common_support,
):
    dftestlog = log_transformer(dftest,log_transform)
    dfsiglog = log_transformer(dfsig,log_transform)
    dfbkglog = log_transformer(dfbkg,log_transform)

    feature_names = dfsiglog.columns.tolist()

    dftestlogtrim, dfsiglogtrim, dfbkglogtrim = trim_to_common_support([dftestlog,dfsiglog,dfbkglog],feature_names)

    dfsiglogtrim["label"] = 1.0
    dfbkglogtrim["label"] = 0.0
    return dfbkglogtrim, dfsiglogtrim, feature_names


@app.cell
def _(dfbkglogtrim, dfsiglogtrim):
    dfsig_val, dfsig_train = split_dataframe(dfsiglogtrim,0.8)
    dfbkg_val, dfbkg_train = split_dataframe(dfbkglogtrim,0.8)
    return dfbkg_train, dfbkg_val, dfsig_train, dfsig_val


@app.cell
def _(dfbkg_train, dfbkg_val, dfsig_train, dfsig_val, pd):
    df_train = pd.concat([dfsig_train, dfbkg_train], ignore_index=True)
    df_test = pd.concat([dfsig_val, dfbkg_val], ignore_index=True)
    return df_test, df_train


@app.cell
def _():
    #feature_names = #Restricting features
    return


@app.cell
def _(df_test, df_train, feature_names):
    X_train_df = df_train[feature_names]

    y_train_df = df_train["label"]

    X_test_df = df_test[feature_names]

    y_test_df = df_test["label"]
    return X_test_df, X_train_df, y_test_df, y_train_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 1: Train a neural network

    Try yourself to setup a "simple" deep NN with a few dense layers that does the classification task.

    Hint: define the model, the setup, the array of losses as global variables, and train for a small number of epochs; if you re-evaluate the training cell (Shift+Enter), the next epochs will be trained. In this way, you get a fast response if your setup has the chance to converge to something useful. For the plot to update correctly, you also need to re-evaluate the plot cell. You can also change the number of epochs to a larger number, once you're happy with the setup.

    Reducing complexity will also help to get started, for instance by reducing the number of input features.

    Learn about NN architectures in the flux documentation. For example about [layers](https://fluxml.ai/Flux.jl/stable/reference/models/layers/), [activation functions](https://fluxml.ai/Flux.jl/stable/reference/models/activation/), [optimisers](https://fluxml.ai/Optimisers.jl/dev/api/) or [losses](https://fluxml.ai/Flux.jl/stable/reference/models/losses/).

    You may also want to ask your trusted AI chatbot for recommendations on the architecture.
    """)
    return


@app.cell
def _(X_test_df, X_train_df, torch, y_test_df, y_train_df):
    X_train = torch.tensor(
        X_train_df.values,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train_df.values,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test_df.values,
        dtype=torch.float32
    )

    y_test = torch.tensor(
        y_test_df.values,
        dtype=torch.float32
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(feature_names, nn):
    model = nn.Sequential(
        nn.Linear(len(feature_names), 16),
        nn.ReLU(),

        nn.Linear(16, 16),
        nn.ReLU(),

        nn.Linear(16, 1)
    )
    return (model,)


@app.cell
def _(model, nn, optim):
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.01
    )
    return loss_fn, optimizer


@app.cell
def _():
    losses = []

    # Number of epochs per execution of the training cell
    epochs = 10
    return epochs, losses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Training cell
    This is the cell you repeatedly execute.
    """)
    return


@app.cell
def _(X_train, epochs, loss_fn, losses, model, optimizer, y_train):
    for epoch in range(epochs):

        # Make predictions
        predictions = model(X_train)

        # Calculate training loss
        loss = loss_fn(predictions, y_train)

        # Reset gradients
        optimizer.zero_grad()

        # Calculate gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Store loss
        losses.append(loss.item())


    print(
        f"Trained for {epochs} more epochs. "
        f"Current loss: {losses[-1]:.4f}"
    )
    return (loss,)


@app.cell
def _(losses, plt):
    plt.figure(figsize=(8, 5))

    plt.plot(
        losses,
        color="blue"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.grid()
    plt.show()
    return


@app.cell
def _(X_test, loss_fn, model, torch, y_test):
    with torch.no_grad():

        test_predictions = model(X_test)

        test_loss = loss_fn(
            test_predictions,
            y_test
        )

    print(f"Test loss: {test_loss.item():.4f}")
    return


@app.cell
def _(X_test, model, test_df, torch):
    with torch.no_grad():

        predictions = model(X_test).numpy().flatten()

    test_df["prediction"] = predictions

    print(test_df.head(10))
    return


@app.cell
def _(plt, test_df):
    plt.figure(figsize=(7, 7))

    plt.scatter(
        test_df["target"],
        test_df["prediction"],
        alpha=0.7
    )

    # Perfect prediction line
    minimum = min(
        test_df["target"].min(),
        test_df["prediction"].min()
    )

    maximum = max(
        test_df["target"].max(),
        test_df["prediction"].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        color="red",
        linestyle="--"
    )

    plt.xlabel("Actual target")
    plt.ylabel("Predicted target")
    plt.title("Actual vs. Predicted")

    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2: Bias-Variance tradeoff

    In this exercise, we use bootstrapped training data to estimate bias and variance of the NN training.

    We first reduce the training sample size to show effects of overtraining, and to be able to train NNs on the bootstrapped datasets faster.
                                                                                                          Your first task for this exercise: Try to build a NN that overtrains.
    """)
    return


@app.cell
def _(df_test, df_train, feature_names):
    X_train_s = df_train[feature_names][:100]

    y_train_s = df_train["label"][:100]

    X_test_s = df_test[feature_names][:100]

    y_test_s = df_test["label"][:100]
    return


@app.cell
def _(feature_names, nn):
    model_overtrain = nn.Sequential(
        nn.Linear(len(feature_names), 16),
        nn.ReLU(),

        nn.Linear(16, 16),
        nn.ReLU(),

        nn.Linear(16, 32),
        nn.ReLU(),

        nn.Linear(32, 16),
        nn.ReLU(),

        nn.Linear(16, 1)
    )

    losses_o = []
    return losses_o, model_overtrain


@app.cell
def _(
    X_train,
    epochs,
    loss,
    loss_fn,
    losses_o,
    model_overtrain,
    optimizer,
    y_train,
):
    for epoch in range(epochs):

        # Make predictions
        predictions_o = model_overtrain(X_train)

        # Calculate training loss
        loss_o = loss_fn(predictions_o, y_train)

        # Reset gradients
        optimizer.zero_grad()

        # Calculate gradients
        loss_o.backward()

        # Update weights
        optimizer.step()

        # Store loss
        losses_o.append(loss.item())


    print(
        f"Trained for {epochs} more epochs. "
        f"Current loss: {losses_o[-1]:.4f}"
    )
    return


@app.cell
def _(losses, plt):
    plt.figure(figsize=(8, 5))

    plt.plot(
        losses,
        color="blue"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2.5: Regularisation
    You will now use the overtrained model and explore regularisation techniques.

    The number of bootstrap samples and training epochs is preset to 100 and 120. As a proxy to measure overtraining, we use the binary cross-entropy of the validation sample in the last epoch. Which other proxies can you think of? What would you compute if you'd be using the test sample?

    One of the conceptually easiest regularisation methods are "Dropout layers" that you can add to your network architecture. Explore how the BCE plot changes with adding dropout or other regularisations.
    """)
    return


@app.cell
def _(StandardScaler, np, tf):
    def bootstrap_overtraining(
        train_df,
        val_df,
        target_col,
        n_epochs=120,
        n_bootstraps=100,
        hidden_layers=(32, 16),
        learning_rate=0.001,
        random_state=42
    ):
        """
        Bootstrap estimate of neural-network overtraining.

        The proxy for overtraining is the binary cross-entropy (BCE)
        on the validation sample at the final epoch.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Training data containing predictors and target.
        val_df : pandas.DataFrame
            Validation data containing predictors and target.
        target_col : str
            Name of the binary target column.
        n_epochs : int
            Number of training epochs. Default = 120.
        n_bootstraps : int
            Number of bootstrap samples. Default = 100.
        hidden_layers : tuple
            Number of neurons in each hidden layer.
        learning_rate : float
            Learning rate for Adam.
        random_state : int
            Random seed.

        Returns
        -------
        bce_values : numpy.ndarray
            Validation BCE at epoch 120 for each bootstrap.
        """

        # ---------------------------------------------------------
        # Separate predictors and target
        # ---------------------------------------------------------
        X_train = train_df.drop(columns=target_col)
        y_train = train_df[target_col]

        X_val = val_df.drop(columns=target_col)
        y_val = val_df[target_col]

        # Convert to numpy
        X_train = X_train.to_numpy(dtype=np.float32)
        y_train = y_train.to_numpy(dtype=np.float32)

        X_val = X_val.to_numpy(dtype=np.float32)
        y_val = y_val.to_numpy(dtype=np.float32)

        # ---------------------------------------------------------
        # Scale predictors
        # Fit scaler on the original training data only
        # ---------------------------------------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        rng = np.random.default_rng(random_state)

        bce_values = []

        # ---------------------------------------------------------
        # Bootstrap loop
        # ---------------------------------------------------------
        for b in range(n_bootstraps):

            # Bootstrap sample of training observations
            indices = rng.integers(
                low=0,
                high=len(X_train_scaled),
                size=len(X_train_scaled)
            )

            X_boot = X_train_scaled[indices]
            y_boot = y_train[indices]

            # -----------------------------------------------------
            # Create a fresh neural network
            # -----------------------------------------------------
            model = tf.keras.Sequential()

            # Input + hidden layers
            for i, units in enumerate(hidden_layers):
                if i == 0:
                    model.add(
                        tf.keras.layers.Dense(
                            units,
                            activation="relu",
                            input_shape=(X_train_scaled.shape[1],)
                        )
                    )
                else:
                    model.add(
                        tf.keras.layers.Dense(
                            units,
                            activation="relu"
                        )
                    )

            # Binary output
            model.add(
                tf.keras.layers.Dense(
                    1,
                    activation="sigmoid"
                )
            )

            # -----------------------------------------------------
            # Compile
            # -----------------------------------------------------
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate
                ),
                loss="binary_crossentropy"
            )

            # -----------------------------------------------------
            # Train for exactly 120 epochs
            # -----------------------------------------------------
            history = model.fit(
                X_boot,
                y_boot,
                epochs=n_epochs,
                batch_size=32,
                verbose=0
            )

            # -----------------------------------------------------
            # BCE on validation sample at final epoch
            # -----------------------------------------------------
            val_bce = model.evaluate(
                X_val_scaled,
                y_val,
                verbose=0
            )

            bce_values.append(val_bce)

            print(
                f"Bootstrap {b + 1:3d}/{n_bootstraps}: "
                f"validation BCE = {val_bce:.4f}"
            )

        return np.array(bce_values)

    

    return (bootstrap_overtraining,)


@app.cell
def _(bootstrap_overtraining, train_df, val_df):
    bce = bootstrap_overtraining(
        train_df=train_df,
        val_df=val_df,
        target_col="label",
        n_epochs=120,
        n_bootstraps=100
    )

    return (bce,)


@app.cell
def _(bce, np):
    print("Mean BCE:", bce.mean())
    print("Median BCE:", np.median(bce))
    print("Std. BCE:", bce.std())

    print(
        "95% bootstrap interval:",
        np.percentile(bce, [2.5, 97.5])
    )
    return


@app.cell
def _(bce, plt):
    plt.hist(bce, bins=20, edgecolor="black")
    plt.xlabel("Validation binary cross-entropy at epoch 120")
    plt.ylabel("Number of bootstrap samples")
    plt.title("Bootstrap distribution of validation BCE")
    plt.show()

    return


if __name__ == "__main__":
    app.run()
